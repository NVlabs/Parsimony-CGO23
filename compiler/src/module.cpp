/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/VectorUtils.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Scalar/StructurizeCFG.h>
#include <llvm/Transforms/Scalar/Scalarizer.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>
#include <llvm/Transforms/Utils/LowerInvoke.h>
#include <llvm/Transforms/Utils/LowerSwitch.h>
#include <llvm/Transforms/Utils/UnifyFunctionExitNodes.h>
#include <llvm/Transforms/Utils/UnifyLoopExits.h>

#include <unordered_set>

#include "diagnostics.h"
#include "function.h"
#include "module.h"
#include "rename_values.h"
#include "utils.h"

using namespace llvm;

namespace ps {

unsigned module_verbosity_level;
[[maybe_unused]] static unsigned& verbosity_level = module_verbosity_level;

Function* ModuleVectorizer::createVectorFunction(Function* F, VFABI& vfabi) {
    PRINT_HIGH("Cloning scalar function " << *F << " with VFABI "
                                          << vfabi.toString());

    // Vectorize the argument types
    std::vector<Type*> arg_types;

    unsigned i = 0;
    for (auto& a : F->args()) {
        assert(i < vfabi.parameters.size());
        if (vfabi.parameters[i].is_varying) {
            arg_types.push_back(vectorizeType(a.getType(), vfabi.vlen));
        } else {
            arg_types.push_back(a.getType());
        }
        i++;
    }
    assert(i == vfabi.parameters.size());

    // Add the mask argument if necessary
    if (vfabi.mask) {
        assert(!vfabi.is_entry_point);
        arg_types.push_back(
            vectorizeType(Type::getInt1Ty(F->getContext()), vfabi.vlen));
    }

    // Extra arguments for the parsim calling convention
    if (vfabi.is_declare_spmd) {
        // Gang num
        arg_types.push_back(Type::getInt64Ty(F->getContext()));

        // Grid size
        arg_types.push_back(Type::getInt64Ty(F->getContext()));
    }

    // Vectorize the return type
    Type* return_type = vectorizeType(F->getReturnType(), vfabi.vlen);
    FunctionType* VT = FunctionType::get(return_type, arg_types, false);

    // Create the function declaration
    Function* VF = Function::Create(VT, F->getLinkage(), vfabi.mangled_name,
                                    F->getParent());
    VF->setCallingConv(F->getCallingConv());

    PRINT_MID("Generated vector function declaration:\n" << *VF);

    // Clone the function body
    // this non-obvious code block derived from the LLVM implementation of
    // CloneFunction()
    ValueToValueMapTy value_map;
    Function::arg_iterator arg = VF->arg_begin();
    for (const Argument& i : F->args()) {
        arg->setName(i.getName());
        value_map[&i] = &*arg;
        arg++;
    }
    SmallVector<ReturnInst*, 4> returns;
    CloneFunctionInto(VF, F, value_map, CloneFunctionChangeType::GlobalChanges,
                      returns);
    // OptimizeNone == -O0, which is incompatible with alwaysinline
    if (!VF->hasFnAttribute(Attribute::OptimizeNone)) {
        VF->removeFnAttr(Attribute::NoInline);
        VF->addFnAttr(Attribute::AlwaysInline);
    }

    PRINT_HIGH("Generated vector function " << VF->getName() << "\n" << *VF);

    return VF;
}

void ModuleVectorizer::setGridGangNum(CallInst* call,
                                      GridMetadata& grid_metadata) {
    Value* op = call->getOperand(0);
    if (grid_metadata.gang_num != 0) {
        FATAL(
            "Found more than one __psim_set_gang_num() call "
            "preceding a call to __kmpc_fork_call: "
            << *call);
    }
    grid_metadata.gang_num = op;
    grid_metadata.populated = true;

    PRINT_HIGH("Set grid gang num to " << *grid_metadata.gang_num);
}

void ModuleVectorizer::setGridGangSize(CallInst* call,
                                       GridMetadata& grid_metadata) {
    ConstantInt* op = dyn_cast<ConstantInt>(call->getOperand(0));
    if (!op) {
        FATAL(
            "Expected ConstantInt argument to "
            "__psim_set_gang_size; but received "
            << *op << "\n");
    }
    if (grid_metadata.vfabi.vlen != 0) {
        FATAL(
            "Found more than one __psim_set_gang_size() call "
            "preceding a call to __kmpc_fork_call: "
            << *call);
    }
    grid_metadata.vfabi.vlen = op->getZExtValue();
    grid_metadata.populated = true;

    PRINT_HIGH("Set grid gang size to " << grid_metadata.vfabi.vlen);
}

void ModuleVectorizer::setGridSize(CallInst* call,
                                   GridMetadata& grid_metadata) {
    Value* op = call->getOperand(0);
    if (grid_metadata.grid_size) {
        FATAL(
            "Found more than one __psim_set_grid_size() "
            "call preceding a call to __kmpc_fork_call: "
            << *call);
    }
    grid_metadata.grid_size = op;
    grid_metadata.populated = true;

    PRINT_HIGH("Set grid grid size to " << *op);
}

void ModuleVectorizer::setGridSubName(CallInst* call,
                                      GridMetadata& grid_metadata) {
    //Value* op = call->getOperand(0);
    if (!grid_metadata.subname.empty()) {
        FATAL(
            "Found more than one __psim_set_grid_sub_name() "
            "call preceding a call to __kmpc_fork_call: "
            << *call);
    }

    // FIXME
#if 0
    Value* op = call->getOperand(0);
    Constant* pCE = dyn_cast<Constant>(op);
    assert(pCE);
    Value* firstop = pCE->getOperand(0);
    assert(firstop);
    PRINT_ALWAYS(*firstop);

    ConstantData* x = dyn_cast<ConstantData>(firstop);
    assert(x);
    PRINT_ALWAYS(*x);


    GlobalVariable* GV = dyn_cast<GlobalVariable>(firstop);
    assert(GV);
    Constant* c = GV->getInitializer();
    assert(c);
    ConstantDataSequential* cds = dyn_cast<ConstantDataSequential>(c);
    assert(cds);
#endif
    grid_metadata.subname = "foo";//cds->getAsCString().str();
    grid_metadata.populated = true;

    PRINT_HIGH("Set grid sub name to " << grid_metadata.subname);
}

void ModuleVectorizer::setGridOmpFunction(CallInst* call,
                                          GridMetadata& grid_metadata) {
    Value* omp_func_value = call->getOperand(2);

    BitCastOperator* bitcast = dyn_cast<BitCastOperator>(omp_func_value);
    if (bitcast) {
        omp_func_value = bitcast->getOperand(0);
    }

    assert(!grid_metadata.omp_func);

    grid_metadata.omp_func = dyn_cast<Function>(omp_func_value);
    if (!grid_metadata.omp_func) {
        FATAL("omp function is not a function? " << *call);
    }
    std::string new_name =
        call->getFunction()->getName().str() + "." + grid_metadata.subname;
    grid_metadata.omp_func->setName(new_name);

    PRINT_LOW("Found psim entry point"
              << grid_metadata.omp_func->getName());
}

void ModuleVectorizer::finishGridMetadata(GridMetadata& grid_metadata) {
    grid_metadata.vfabi.is_entry_point = true;
    grid_metadata.vfabi.is_declare_spmd = true;

    uint32_t num_lanes = grid_metadata.vfabi.vlen;
    // If grid size wasn't explicitly set, then set it to match gang size
    assert(num_lanes != 0);
    if (grid_metadata.grid_size == 0) {
        grid_metadata.grid_size =
            ConstantInt::get(Type::getInt64Ty(vm_info.ctx), num_lanes);
    }

    // gang num is also implicitly 0 if it wasn't already set...no need to write
    // code for if (gang_num == 0) { gang_num = 0; }

    grid_metadata.vfabi.isa = "e";  // FIXME
    grid_metadata.vfabi.mask = false;
    for (unsigned i = 0; i < grid_metadata.omp_func->arg_size(); i++) {
        // TODO extract alignment from fork_call arguments
        grid_metadata.vfabi.parameters.push_back(VFABIShape::Uniform());
    }

    grid_metadata.vfabi.scalar_name = grid_metadata.omp_func->getName();
    grid_metadata.vfabi.mangled_name = grid_metadata.vfabi.toString();
}

void ModuleVectorizer::findPsimCalls(
    std::unordered_map<CallInst*, GridMetadata>& grids,
    std::unordered_set<CallInst*>& insts_to_delete) {
    for (Function& F : vm_info.mod->functions()) {
        for (BasicBlock& BB : F) {
            // Make sure every use of psim APIs is followed by a call to
            // __kmpc_fork_call within the same basic block
            GridMetadata grid_metadata;

            for (Instruction& I : BB) {
                CallInst* call = dyn_cast<CallInst>(&I);
                if (!call) {
                    continue;
                }

                Function* called_function = call->getCalledFunction();
                if (!called_function) {
                    continue;
                }
                StringRef name = called_function->getName();

                if (name == "__psim_set_gang_num") {
                    setGridGangNum(call, grid_metadata);
                    insts_to_delete.insert(call);
                } else if (name == "__psim_set_gang_size") {
                    setGridGangSize(call, grid_metadata);
                    insts_to_delete.insert(call);
                } else if (name == "__psim_set_grid_size") {
                    setGridSize(call, grid_metadata);
                    insts_to_delete.insert(call);
                } else if (name == "__psim_set_grid_sub_name") {
                    setGridSubName(call, grid_metadata);
                    insts_to_delete.insert(call);
                } else if (name == "__kmpc_fork_call") {
                    PRINT_HIGH("Found call to __kmpc_fork_call: " << I);
                    if (!grid_metadata.populated) {
                        continue;
                    }
                    setGridOmpFunction(call, grid_metadata);
                    finishGridMetadata(grid_metadata);
                    grids.insert(std::make_pair(call, grid_metadata));

                    // Reset grid_metadata;
                    grid_metadata = GridMetadata();
                }
            }

            if (grid_metadata.populated) {
                FATAL(
                    "Grid metadata not followed by call to __kmpc_fork_call "
                    "within the same basic block: "
                    << BB.getName() << "\n");
            }
        }
    }
}

void ModuleVectorizer::insertPsimGrids(
    std::unordered_map<CallInst*, GridMetadata>& grids) {
    // Replace all entry point call instructions to __kmpc_fork_call with
    // calls directly to the omp function (which will be vectorized)
    for (auto i : grids) {
        CallInst* call = i.first;
        GridMetadata grid_metadata = i.second;

        PRINT_HIGH("Inserting grid for " << *call);

        // Replace the call to __kmpc_fork_call with a call to the
        // to-be-vectorized entry point function
        std::vector<Value*> new_args;

        // Push back two dummy i32* nullptrs to match the omp parallel calling
        // convention.  These are just 0) debug info and 1) num arguments, so
        // we don't really need them for anything
        for (unsigned i = 0; i < 2; i++) {
            new_args.push_back(ConstantPointerNull::get(
                PointerType::get(Type::getInt32Ty(vm_info.ctx), 0)));
        }

        // Push back the actual operands
        for (unsigned i = 3; i < call->getNumOperands() - 1; i++) {
            new_args.push_back(call->getOperand(i));
        }

        // There shouldn't be a mask argument
        assert(!grid_metadata.vfabi.mask);

        // Add the parsim-specific arguments
        if (grid_metadata.vfabi.is_declare_spmd) {
            new_args.push_back(grid_metadata.gang_num);
            PRINT_HIGH("Adding gang num argument " << *new_args.back());
            new_args.push_back(grid_metadata.grid_size);
            PRINT_HIGH("Adding grid size argument " << *new_args.back());
        }

        // Create the new call, and replace the original call to
        // __kmpc_fork_call
        CallInst* new_call =
            CallInst::Create(grid_metadata.omp_func->getFunctionType(),
                             grid_metadata.omp_func, new_args, call->getName());
        ReplaceInstWithInst(call, new_call);

        entry_points.insert(
            std::make_pair(grid_metadata.omp_func, grid_metadata.vfabi));
    }
}

void ModuleVectorizer::findPSVEntryPoints() {
    // We can't delete instructions while iterating over them, so store them in
    // these two structures and delete/replace them after iterating
    std::unordered_map<CallInst*, GridMetadata> grids;
    std::unordered_set<CallInst*> insts_to_delete;

    findPsimCalls(grids, insts_to_delete);

    for (Instruction* I : insts_to_delete) {
        I->eraseFromParent();
    }

    insertPsimGrids(grids);
}

void ModuleVectorizer::replaceUnreachableInsts(Function* F) {
    PRINT_MID("Replacing unreachable instructions");

    Type* return_type = F->getFunctionType()->getReturnType();
    PRINT_HIGH("Return type is " << *return_type);

    Value* ret_val = nullptr;
    if (!return_type->isVoidTy()) {
        // Create a dummy return value of the appropriate type
        ret_val = llvm::Constant::getNullValue(return_type);
        PRINT_HIGH("Return value is " << *ret_val);
    }

    for (BasicBlock& BB : *F) {
        Instruction* term = BB.getTerminator();
        UnreachableInst* I = dyn_cast<UnreachableInst>(term);
        if (!I) {
            continue;
        }

        // Replace I with a return of the appropriate type
        ReturnInst* ret = ReturnInst::Create(F->getContext(), ret_val);
        ReplaceInstWithInst(I, ret);
    }
}

void ModuleVectorizer::preprocessFunction(Function* F) {
    // Built-in passes
    FunctionAnalysisManager FAM;

    PassBuilder PB;
    PB.registerFunctionAnalyses(FAM);

    // The order of the optimization passes below matters!

    // Step 1: remove invoke instructions; exceptions that are triggered won't
    // be able to be caught!
    FunctionPassManager FPM;
    FPM.addPass(LowerInvokePass());
    FPM.addPass(SimplifyCFGPass());
    FPM.run(*F, FAM);

/*    PRINT_ALWAYS("Scalarizing function " << F->getName());
    PRINT_ALWAYS(*F);*/
    auto scalarizer = ScalarizerPass();
    scalarizer.setScalarizeVariableInsertExtract(true);
    scalarizer.setScalarizeLoadStore(true);
    scalarizer.run(*F,FAM);

/*    PRINT_ALWAYS("Finished scalarizing function " << F->getName());
    PRINT_ALWAYS(*F);*/

    // Step 2: replace unreachable instructions with return instructions so that
    // we can put the code into a form suitable for StructurizeCFG
    replaceUnreachableInsts(F);

    // Step 3: more built-in passes to put the code into a form suitable for
    // StructurizeCFG
    FunctionPassManager FPM1;
    FPM1.addPass(UnifyFunctionExitNodesPass());
    FPM1.addPass(LowerSwitchPass());
    FPM1.addPass(LoopSimplifyPass());
    FPM1.addPass(UnifyLoopExitsPass());
    FPM1.addPass(StructurizeCFGPass());
    FPM1.run(*F, FAM);

    renameValues(*F);
}

void ModuleVectorizer::initialize() {
    findPSVEntryPoints();

    // Store the list of original functions in a vector so that we don't try
    // to recursively analyze functions we've generated and which have been
    // added to the end of the list of functions in the module
    std::vector<Function*> functions;
    for (Function& F : vm_info.mod->functions()) {
        functions.push_back(&F);
    }

    for (Function* F : functions) {
        PRINT_LOW("Analyzing function " << F->getName());
        std::vector<ps::VFABI> vfabis;

        auto i = entry_points.find(F);
        if (i != entry_points.end()) {
            vfabis.push_back(i->second);
        } else {
            getFunctionVFABIs(F, vfabis);
        }

        if (vfabis.empty()) {
            PRINT_HIGH("No VFABIs found");
            continue;
        }

        for (ps::VFABI& vfabi : vfabis) {
            PRINT_LOW("Analyzing VFABI \"" << vfabi.mangled_name << "\"");

            Function* VF = createVectorFunction(F, vfabi);
            VectorizedFunctionInfo* vf_info =
                new VectorizedFunctionInfo(vm_info, VF, vfabi);
            vf_info->VF = VF;
            vf_info->vfabi = vfabi;
            vm_info.vfinfo_map[F].push_back(vf_info);

            preprocessFunction(VF);
        }
    }
}

void ModuleVectorizer::vectorizeFunctions() {
    // add vectorized functions (not entry points) to
    // the resolver map, these are vectorized not
    // inlined function calls
    for (auto it : vm_info.vfinfo_map) {
        for (VectorizedFunctionInfo* info : it.second) {
            if (!info->vfabi.is_entry_point) {
                vm_info.function_resolver.add(it.first,
                                              {info->VF, info->vfabi});
            }
        }
    }
    // vectorize all the functions
    for (auto& i : vm_info.vfinfo_map) {
        Function* F = i.first;
        for (VectorizedFunctionInfo* vf_info : i.second) {
            FunctionVectorizer(*vf_info).vectorize();

            if (vf_info->vfabi.is_entry_point) {
                assert(i.second.size() == 1);
                PRINT_LOW("Replacing all uses of " << F->getName() << " with "
                                                   << vf_info->VF->getName());
                F->replaceAllUsesWith(vf_info->VF);
                F->eraseFromParent();
            }

            printDiagnostics(vf_info);
        }
    }
}

void ModuleVectorizer::writeToFile(const std::string& fileName) {
    assert(vm_info.mod);
    std::error_code EC;
    raw_fd_ostream file(fileName, EC, sys::fs::OpenFlags::OF_None);
    vm_info.mod->print(file, nullptr);
    if (EC) {
        FATAL("ERROR: printing module to file failed: " << EC.message());
    }
    file.close();
}

}  // namespace ps
