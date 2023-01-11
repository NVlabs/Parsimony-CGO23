/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "transform.h"

#include <llvm/Analysis/DomTreeUpdater.h>
#include <llvm/Analysis/VectorUtils.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <cassert>
#include <cstdarg>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "resolver.h"
#include "utils.h"
#include "vectorize.h"

using namespace llvm;

namespace ps {

unsigned transform_verbosity_level;
[[maybe_unused]] static unsigned& verbosity_level = transform_verbosity_level;

TransformStep::TransformStep(VectorizedFunctionInfo& vf_info)
    : vf_info(vf_info),
      value_cache(vf_info.value_cache),
      num_lanes(vf_info.num_lanes) {}

Value* TransformStep::transformSimpleInstruction(Instruction* inst) {
    if (value_cache.has(inst) && value_cache.getShape(inst).isVarying()) {
        inst->mutateType(vf_info.vectorizeType(inst->getType()));

        for (unsigned i = 0; i < inst->getNumOperands(); i++) {
            Value* v = value_cache.getVectorValue(inst->getOperand(i));
            inst->setOperand(i, v);
        }

        return inst;
    } else {
        return transformInstructionWithoutVectorizing(inst);
    }
}

Value* TransformStep::transformReturn(ReturnInst* inst) {
    Value* ret_val = inst->getReturnValue();
    if (!ret_val) {
        return transformInstructionWithoutVectorizing(inst);
    }

    if (value_cache.getShape(ret_val).isVarying()) {
        Value* v = value_cache.getVectorValue(inst->getOperand(0));
        inst->setOperand(0, v);
        return inst;
    } else {
        return transformInstructionWithoutVectorizing(inst);
    }
}

Value* TransformStep::transformMemInst(Instruction* inst) {
    MemInstMappedShape minst_shape = value_cache.getMemInstMappedShape(inst);

    PRINT_HIGH("Transforming " << *inst << ": " << minst_shape.toString());
    switch (minst_shape.mapped_shape) {
        case MemInstMappedShape::UNIFORM: {
            return transformInstructionWithoutVectorizing(inst);
        } break;
        case MemInstMappedShape::GLOBAL_VALUE: {
            value_cache.setToBeDeleted(inst);
            return ConstantInt::get(inst->getType(), 0);
        } break;
        case MemInstMappedShape::ALREADY_PACKED:
        case MemInstMappedShape::PACKED: {
            return vectorizeMemInst(inst, true);
        } break;
        case MemInstMappedShape::PACKED_SHUFFLE: {
            return vectorizeMemInst(inst, true, minst_shape.indices,
                                    minst_shape.elem_size);
        } break;
        case MemInstMappedShape::GATHER_SCATTER: {
            size_t esize = minst_shape.elem_size;
            return vectorizeMemInst(inst, false, {}, esize);
        } break;
        default:
            FATAL("unreachable");
            break;
    }
}

void TransformStep::rebaseMemPackedIndices(std::vector<int>& indices,
                                           int& min_index, unsigned& factor) {
    factor = 1;
    min_index = 0;
    if (!indices.empty()) {
        min_index = *std::min_element(indices.begin(), indices.end());
        int max = 0;
        for (auto& i : indices) {
            int idx = i - min_index;
            ASSERT(idx >= 0 && idx < INT_MAX,
                   "PACKED_SHUFFLE index " << i << " out of range");
            max = idx > max ? idx : max;
            i = idx;
        }
        factor = ceilDiv((unsigned)max + 1, num_lanes);
    }
}

Value* TransformStep::generateMaskForMemInst(Instruction* inst,
                                             std::vector<int> indices,
                                             unsigned factor) {
    Value* bb_mask = value_cache.getVectorValue(
        vf_info.bb_masks[inst->getParent()].active_mask);
    if (indices.empty()) {
        return bb_mask;
    }

    IRBuilder<> builder(inst->getParent());
    builder.SetInsertPoint(inst->getNextNode());
    std::string name;
    name = inst->getName().str() + ".";

    std::vector<Constant*> mask_vector;
    mask_vector.resize(num_lanes * factor, builder.getFalse());
    Value* bb_mask_shfl =
        value_cache.genConstVect(ConstantVector::get(mask_vector), builder);

    for (uint64_t v : indices) {
        assert(v < mask_vector.size());
        mask_vector[v] = builder.getTrue();
    }
    Value* idx_mask =
        value_cache.genConstVect(ConstantVector::get(mask_vector), builder);

    bool done = false;
    while (!done) {
        std::vector<int> mask_shfl_indices;
        mask_shfl_indices.resize(num_lanes * factor, -1);
        done = true;
        for (unsigned i = 0; i < num_lanes; i++) {
            int pos = indices[i];
            if (pos == -1) {
                continue;
            }
            done = false;
            assert((size_t)pos < mask_shfl_indices.size());
            if (mask_shfl_indices[pos] == -1) {
                mask_shfl_indices[pos] = i;
                indices[i] = -1;
            }
        }
        if (!done) {
            Value* tmp =
                builder.CreateShuffleVector(bb_mask, mask_shfl_indices, name);
            bb_mask_shfl = builder.CreateOr(bb_mask_shfl, tmp, name);
        }
    }
    return builder.CreateAnd(bb_mask_shfl, idx_mask, name);
}

Value* TransformStep::vectorizeMemInst(Instruction* inst, bool packed,
                                       std::vector<int> indices, size_t esize) {
    LoadInst* ld = dyn_cast<LoadInst>(inst);
    StoreInst* st = dyn_cast<StoreInst>(inst);

    Value* val =
        st ? value_cache.getVectorValue(st->getValueOperand()) : nullptr;
    Value* ptr = st ? st->getPointerOperand() : ld->getPointerOperand();
    uint64_t align_val = st ? st->getAlign().value() : ld->getAlign().value();
    auto align = st ? st->getAlign() : ld->getAlign();
    auto ordering = st ? st->getOrdering() : ld->getOrdering();
    assert(align_val < INT32_MAX);
    assert(ordering == AtomicOrdering::NotAtomic);

    IRBuilder<> builder(inst->getParent());
    builder.SetInsertPoint(inst->getNextNode());

    std::string name;
    name = inst->getName().str() + ".";

    // Rebase indices and get min index value and factor
    unsigned factor = 1;
    int min_index = 0;
    rebaseMemPackedIndices(indices, min_index, factor);

    Type* ty = st ? val->getType() : ld->getType();
    Type* sty = ty->getScalarType();

    Value* mask = generateMaskForMemInst(inst, indices, factor);
    Value* ret;

    Type* vty = VectorType::get(sty, getElementCount(num_lanes * factor));
    if (packed) {
        ptr = value_cache.getScalarValue(ptr);

        // Add min index to the base pointer
        if (min_index != 0) {
            assert(esize != 0);
            Value* offset = builder.getInt64(min_index * esize);
            Value* ptr64 =
                builder.CreatePtrToInt(ptr, builder.getInt64Ty(), name);
            ptr64 = builder.CreateAdd(ptr64, offset, name);
            ptr = builder.CreateIntToPtr(ptr64, ptr->getType(), name);
        }

        // shuffle value only for stores
        if (st && !indices.empty()) {
            std::vector<int> value_shfl_indices;
            value_shfl_indices.resize(num_lanes * factor, 0);
            for (unsigned i = 0; i < num_lanes; i++) {
                assert(indices[i] >= 0);
                assert((size_t)indices[i] < value_shfl_indices.size());
                value_shfl_indices[indices[i]] = i;
            }
            val = builder.CreateShuffleVector(val, value_shfl_indices, name);
        }

        Type* pty = PointerType::get(vty, 0);
        Value* p = builder.CreateBitCast(ptr, pty, name);

        if (st) {
            ret = builder.CreateMaskedStore(val, p, align, mask);
        } else {
            ret = builder.CreateMaskedLoad(vty, p, align, mask, nullptr, name);
        }
        // shuffle result only for load
        if (ld && !indices.empty()) {
            ret = builder.CreateShuffleVector(ret, indices, name);
        }

    } else {
        Value* ptrs = value_cache.getVectorValue(ptr);
        assert(esize != 0);
        printWarning(inst, "scatter/gather emitted");
        if (st) {
            vf_info.diagnostics.scatters[esize].push_back(valueString(inst));
            ret = builder.CreateMaskedScatter(val, ptrs, align, mask);
        } else {
            vf_info.diagnostics.gathers[esize].push_back(valueString(inst));
            ret = builder.CreateMaskedGather(vty, ptrs, align, mask, nullptr,
                                             name);
        }
    }
    value_cache.setToBeDeleted(inst);
    return ret;
}

Value* TransformStep::transformBranch(BranchInst* inst) {
    // For conditional branches, vectorize the condition
    if (inst->isConditional()) {
        Shape shape = value_cache.getShape(inst->getCondition());
        if (!shape.isUniform()) {
            IRBuilder<> builder(inst->getParent());
            builder.SetInsertPoint(inst);
            Value* red = builder.CreateOrReduce(
                value_cache.getVectorValue(inst->getCondition()));
            red->setName(inst->getParent()->getName() + "_any");
            inst->setCondition(red);
        }
    }

    return inst;
}

Value* TransformStep::vectorizeUniformCall(CallInst* inst) {
    PRINT_LOW("Vectorizing call through one uniform call per lane: " << *inst);

    Type* i32 = Type::getInt32Ty(vf_info.ctx);
    Type* ret_type = vf_info.vectorizeType(inst->getType());

    // At the original call point, split the basic block into two pieces
    DomTreeUpdater updater(vf_info.doms, DomTreeUpdater::UpdateStrategy::Eager);
    BasicBlock* old_BB_first_half = inst->getParent();
    BasicBlock* old_BB_second_half =
        SplitBlock(inst->getParent(), inst, &updater, vf_info.loop_info);
    vf_info.bb_masks[old_BB_second_half] = vf_info.bb_masks[old_BB_first_half];

    // Extract the active mask from the original basic block
    assert(vf_info.bb_masks.find(old_BB_first_half) != vf_info.bb_masks.end());
    Value* mask = vf_info.bb_masks[old_BB_first_half].active_mask;
    if (!mask) {
        FATAL("BB " << old_BB_first_half->getName() << " has no mask?");
    }
    if (!mask->getType()->isVectorTy()) {
        mask = value_cache.getVectorValue(mask);
    }

    // Create a return value of the right type
    Value* return_value = nullptr;
    if (ret_type != Type::getVoidTy(inst->getContext())) {
        return_value = UndefValue::get(ret_type);
    }

    // Create a series of new basic blocks, two for each lane
    std::vector<BasicBlock*> new_BBs;
    for (unsigned lane = 0; lane < num_lanes; lane++) {
        BasicBlock* BB_check_mask = BasicBlock::Create(
            vf_info.ctx,
            inst->getName() + "_uniformcall_mask" + std::to_string(lane),
            old_BB_first_half->getParent());
        new_BBs.push_back(BB_check_mask);
    }

    // Redirect the old call instruction to branch to these new BBs instead
    BranchInst* term = cast<BranchInst>(old_BB_first_half->getTerminator());
    term->setSuccessor(0, new_BBs[0]);

    // Fill in the BBs for each lane
    for (unsigned lane = 0; lane < num_lanes; lane++) {
        BasicBlock* BB_check_mask = new_BBs[lane];

        // First BB: check the mask for this lane
        Instruction* mask_lane = ExtractElementInst::Create(
            mask, ConstantInt::get(i32, lane),
            inst->getName() + "_mask_lane" + std::to_string(lane),
            BB_check_mask);

        // Figure out the fall through BB
        BasicBlock* next_BB;
        if (lane == num_lanes - 1) {
            next_BB = old_BB_second_half;
        } else {
            next_BB = new_BBs[lane + 1];
        }

        // Create the active lane BB
        BasicBlock* BB_do_call = BasicBlock::Create(
            vf_info.ctx,
            inst->getName() + "_uniformcall_call" + std::to_string(lane),
            old_BB_first_half->getParent());

        // Branch to the active BB or the fallthrough BB
        BranchInst::Create(BB_do_call, next_BB, mask_lane, BB_check_mask);

        // Extract the uniform argument values from each lane
        std::vector<Value*> uniform_args;
        unsigned arg_id = 0;
        for (Use& arg : inst->args()) {
            // No need to vectorize and then extract if it was just a
            // constant to begin with
            Constant* C = dyn_cast<Constant>(&arg);
            if (C) {
                uniform_args.push_back(C);
            } else {
                uniform_args.push_back(ExtractElementInst::Create(
                    value_cache.getVectorValue(arg.get()),
                    ConstantInt::get(i32, lane),
                    inst->getName() + "_arg" + std::to_string(arg_id) +
                        "_lane" + std::to_string(lane),
                    BB_do_call));
            }
            arg_id++;
        }

        // Create a uniform call
        CallInst* call =
            CallInst::Create(inst->getFunctionType(), inst->getCalledOperand(),
                             uniform_args, "", BB_do_call);
        call->setCallingConv(inst->getCallingConv());
        if (inst->getDebugLoc()) {
            call->setDebugLoc(inst->getDebugLoc());
        }
        if (return_value) {
            call->setName(inst->getName() + "_lane" + std::to_string(lane));
        }

        // Populate the lane of the return value
        if (return_value) {
            Value* old_return_value = return_value;
            Instruction* new_return_value = InsertElementInst::Create(
                return_value, call, ConstantInt::get(i32, lane),
                inst->getName() + "_retval" + std::to_string(lane), BB_do_call);

            PHINode* phi = PHINode::Create(
                ret_type, 2,
                inst->getName() + "_ret_phi" + std::to_string(lane));
            if (next_BB->begin() == next_BB->end()) {
                next_BB->getInstList().push_back(phi);
            } else {
                phi->insertBefore(&next_BB->front());
            }
            phi->addIncoming(old_return_value, BB_check_mask);
            phi->addIncoming(new_return_value, BB_do_call);
            return_value = phi;
        }

        // Branch to the fallthrough BB
        BranchInst::Create(next_BB, BB_do_call);
    }

    // Recalculate the dominator and loop analysis now that we've changed
    // the CFG
    // FIXME: it shouldn't be necessary to recalculate the whole thing
    vf_info.FAM.clear();
    vf_info.getAnalyses();

    value_cache.setToBeDeleted(inst);
    return return_value;
}

Value* TransformStep::transformCall(CallInst* inst) {
    Function* f = inst->getCalledFunction();

    // check if function pointer
    if (!f) {
        vf_info.diagnostics.function_pointer_calls.push_back(valueString(inst));
        return vectorizeUniformCall(inst);
    }

    // check if psim API call
    Value* psim_api = transformCallPsimApi(inst);
    if (psim_api) {
        return psim_api;
    }

    // check if instrinsic
    Value* instrinsic = transformCallIntrinsic(inst);
    if (instrinsic) {
        return instrinsic;
    }

#ifdef SLEEF_ENABLE
    // check if vector math
    Value* vmath = transformCallVmath(inst);
    if (vmath) {
        return vmath;
    }
#endif

    // check if this call is to another vectorized function
    Value* vfunc = transformCallVectFunction(inst);
    if (vfunc) {
        return vfunc;
    }

    ItaniumPartialDemangler demangler;
    bool error = demangler.partialDemangle(f->getName().str().c_str());
    std::string dname;
    if (error) {
        dname = f->getName().str();
    } else {
        dname = demangler.finishDemangle(nullptr, nullptr);
    }
    if (dname.find("ostream") == std::string::npos &&
        dname.find("print") == std::string::npos &&
        dname.find("fflush") == std::string::npos &&
        dname.find("assert") == std::string::npos) {
        printWarning(inst, "scalarized function call " + dname);
    }
    vf_info.diagnostics.scalarized_called_functions.insert(f->getName().str());
    return vectorizeUniformCall(inst);
}

Value* TransformStep::transformCallPsimApi(llvm::CallInst* inst) {
    Function* f = inst->getCalledFunction();

    FunctionResolver::PsimApiEnum api_enum =
        vf_info.vm_info.function_resolver.getPsimApiEnum(f);
    if (api_enum == FunctionResolver::PsimApiEnum::PSIM_API_NONE) {
        return nullptr;
    }

    Type* i8 = Type::getInt8Ty(inst->getContext());
    Type* i16 = Type::getInt16Ty(inst->getContext());
    Type* i32 = Type::getInt32Ty(inst->getContext());
    Type* f32 = Type::getFloatTy(inst->getContext());
    Type* i64 = Type::getInt64Ty(inst->getContext());

    IRBuilder<> builder(inst->getParent());
    builder.SetInsertPoint(inst->getNextNode());

    std::string name;
    name = inst->getName().str() + ".";

    switch (api_enum) {
        case FunctionResolver::PsimApiEnum::GET_LANE_NUM: {
            value_cache.setToBeDeleted(inst);
            return ConstantInt::get(i32, 0);
        }

        case FunctionResolver::PsimApiEnum::GET_GANG_SIZE: {
            value_cache.setToBeDeleted(inst);
            return ConstantInt::get(i32, num_lanes);
        }
        case FunctionResolver::PsimApiEnum::GET_GANG_NUM: {
            assert(vf_info.vfabi.is_declare_spmd);
            value_cache.setToBeDeleted(inst);
            return vf_info.VF->getArg(vf_info.VF->arg_size() - 2);
        }
        case FunctionResolver::PsimApiEnum::GET_GRID_SIZE: {
            assert(vf_info.vfabi.is_declare_spmd);
            value_cache.setToBeDeleted(inst);
            return vf_info.VF->getArg(vf_info.VF->arg_size() - 1);
        }
        case FunctionResolver::PsimApiEnum::GET_THREAD_NUM: {
            assert(vf_info.vfabi.is_declare_spmd);
            Value* gang_num = vf_info.VF->getArg(vf_info.VF->arg_size() - 2);
            Value* gang_size = ConstantInt::get(i64, num_lanes);
            Value* base_tnum = builder.CreateMul(gang_num, gang_size, name);
            value_cache.setToBeDeleted(inst);
            return base_tnum;
        }
        case FunctionResolver::PsimApiEnum::GET_OMP_THREAD_NUM: {
            return inst;
        }
        case FunctionResolver::PsimApiEnum::UADD_SAT:
        case FunctionResolver::PsimApiEnum::SADD_SAT:
        case FunctionResolver::PsimApiEnum::USUB_SAT:
        case FunctionResolver::PsimApiEnum::SSUB_SAT: {
            llvm::Intrinsic::ID id =
                vf_info.vm_info.function_resolver.LlvmInstrinsicMap[api_enum];

            Type* ty = f->getFunctionType()->getReturnType();
            if (value_cache.getShape(inst).isVarying()) {
                ty = VectorType::get(ty, getElementCount(num_lanes));
            }

            Function* intrinsic =
                Intrinsic::getDeclaration(inst->getModule(), id, {ty});

            SmallVector<Value*> args = generateArgsForIntrinsics(inst);

            Value* ret = builder.CreateCall(intrinsic, args);
            value_cache.setToBeDeleted(inst);
            return ret;
        }
        case FunctionResolver::PsimApiEnum::ZIP_SYNC: {
            Type* ret_ty = f->getFunctionType()->getReturnType();
            TypeSize ret_ts =
                vf_info.data_layout.getTypeAllocSize(ret_ty->getScalarType());
            Value* in = value_cache.getVectorValue(inst->getOperand(0));
            Type* in_scalar_type = in->getType()->getScalarType();
            TypeSize in_ts =
                vf_info.data_layout.getTypeAllocSize(in_scalar_type);
            unsigned factor = ret_ts.getFixedSize() / in_ts.getFixedSize();
            assert(num_lanes % factor == 0);
            int new_num_lanes = num_lanes / factor;
            ret_ty = VectorType::get(ret_ty, getElementCount(new_num_lanes));
            Value* ret = builder.CreateBitCast(in, ret_ty);
            std::vector<Value*> vectors;
            for (unsigned i = 0; i < factor; i++) {
                vectors.push_back(ret);
            }
            ret_ty = VectorType::get(f->getFunctionType()->getReturnType(),
                                     getElementCount(num_lanes));
            Value* concat = concatenateVectors(builder, vectors);
            ret = builder.CreateBitCast(
                concat,
                VectorType::get(f->getFunctionType()->getReturnType(),
                                getElementCount(num_lanes)),
                name);

            value_cache.setToBeDeleted(inst);
            return ret;
        }

        case FunctionResolver::PsimApiEnum::UNZIP_SYNC: {
            Type* ret_ty = f->getFunctionType()->getReturnType();
            TypeSize ret_ts =
                vf_info.data_layout.getTypeAllocSize(ret_ty->getScalarType());
            Value* in = value_cache.getVectorValue(inst->getOperand(0));
            ConstantInt* idx = dyn_cast<ConstantInt>(inst->getOperand(1));
            assert(idx);
            unsigned index = idx->getZExtValue();
            Type* in_scalar_type = in->getType()->getScalarType();
            TypeSize in_ts =
                vf_info.data_layout.getTypeAllocSize(in_scalar_type);
            unsigned factor = in_ts.getFixedSize() / ret_ts.getFixedSize();
            assert(index < factor);
            Type* vty = VectorType::get(in_scalar_type,
                                        getElementCount(num_lanes / factor));
            index = index * num_lanes / factor;
            Value* sa = builder.CreateExtractVector(
                vty, in, ConstantInt::get(i64, index), name);

            ret_ty = VectorType::get(ret_ty, getElementCount(num_lanes));
            Value* ret = builder.CreateBitCast(sa, ret_ty);

            value_cache.setToBeDeleted(inst);
            return ret;
        }

        case FunctionResolver::PsimApiEnum::SHFL_SYNC: {
            ItaniumPartialDemangler demangler;
            bool error = demangler.partialDemangle(f->getName().str().c_str());
            if (error) {
                FATAL("Could not demangle " << f->getName());
            }
            char* ret_type_c_str =
                demangler.getFunctionReturnType(nullptr, nullptr);
            std::string ret_type_str(ret_type_c_str);
            free(ret_type_c_str);

            llvm::Type* ret_type = nullptr;
            bool is_unsigned;
            if (ret_type_str == "unsigned char") {
                ret_type = i8;
                is_unsigned = true;
            } else if (ret_type_str == "char") {
                ret_type = i8;
                is_unsigned = false;
            } else if (ret_type_str == "signed char") {
                ret_type = i8;
                is_unsigned = false;
            } else if (ret_type_str == "unsigned int") {
                ret_type = i32;
                is_unsigned = true;
            } else if (ret_type_str == "int") {
                ret_type = i32;
                is_unsigned = false;
            } else if (ret_type_str == "float") {
                ret_type = f32;
                is_unsigned = false;
            } else if (ret_type_str == "unsigned short") {
                ret_type = i16;
                is_unsigned = true;
            } else {
                FATAL("SHFL " << f->getName() << " " << ret_type_str);
            }

            int num_value_operands = inst->getNumOperands() - 2;
            assert(num_value_operands == 1 || num_value_operands == 2);

            /*    Get operands a, b, and shuffle index  */
            Value* va = value_cache.getVectorValue(inst->getOperand(0));
            Type* vty = VectorType::get(va->getType()->getScalarType(),
                                        getElementCount(num_lanes));
            Value* zero = Constant::getNullValue(vty);

            Value* vb = num_value_operands == 1
                            ? zero
                            : value_cache.getVectorValue(inst->getOperand(1));
            Value* idx = num_value_operands == 1 ? inst->getOperand(1)
                                                 : inst->getOperand(2);

            ///////

            Shape sidx = value_cache.getShape(idx);
            PRINT_HIGH("Shuffle pattern is " << sidx.toString());
            ASSERT(sidx.isIndexed() && sidx.hasConstantBase(),
                   "shuffle pattern cannot be reduced to const operation "
                       << *idx << " " << sidx.toString());

            std::vector<Constant*> vidxs;
            for (int64_t i = 0; i < num_lanes; i++) {
                uint64_t idx = sidx.getValueAtLane(i);
                if (idx >= 0 && idx < num_lanes * num_value_operands) {
                    vidxs.push_back(builder.getInt32(idx));
                } else {
                    // Results in zero value
                    vidxs.push_back(
                        builder.getInt32(static_cast<int32_t>(num_lanes + i)));
                }
            }
            Value* idxs = ConstantVector::get(vidxs);
            PRINT_HIGH("shuffle Idx " << *idxs);

            unsigned from_bits = va->getType()->getScalarSizeInBits();
            unsigned to_bits = ret_type->getScalarSizeInBits();
            unsigned bits_ratio = to_bits / from_bits;

            PRINT_HIGH("For shuffle: " << *inst);
            PRINT_HIGH("Return type: " << *ret_type);
            PRINT_HIGH("is_unsigned: " << is_unsigned);
            PRINT_HIGH("from_bits: " << from_bits);
            PRINT_HIGH("to_bits: " << to_bits);
            PRINT_HIGH("bits_ratio: " << bits_ratio);

            Value* shfl = nullptr;
            if (from_bits == to_bits || !is_unsigned ||
                from_bits * bits_ratio != to_bits) {
                shfl = builder.CreateShuffleVector(va, vb, idxs, name);
            }

            Value* ret;
            if (from_bits == to_bits) {
                PRINT_HIGH("from_bits == to_bits");
                ret = shfl;
            } else if (is_unsigned && from_bits * bits_ratio == to_bits) {
                PRINT_HIGH("zero extend and shuffle");
                // Built-in zero extend
                std::vector<Value*> vectors;

                std::vector<int> new_indices;
                for (unsigned i = 0; i < num_lanes; i++) {
                    // Assumes little-endianness!
                    uint64_t index = sidx.getValueAtLane(i);
                    if (index >= UINT_MAX) {
                        new_indices.push_back(num_lanes);
                    } else {
                        new_indices.push_back(index);
                    }
                    for (unsigned j = 1; j < bits_ratio; j++) {
                        // position 'num_lanes' is the first element of the
                        // second argument, which is the variable zero
                        new_indices.push_back(num_lanes);
                    }

                    if (new_indices.size() == num_lanes) {
                        vectors.push_back(builder.CreateShuffleVector(
                            va, zero, new_indices, name));
                        new_indices.clear();
                    }
                }
                assert(new_indices.empty());

                Value* concat = concatenateVectors(builder, vectors);
                ret = builder.CreateBitCast(
                    concat,
                    VectorType::get(inst->getType()->getScalarType(),
                                    getElementCount(num_lanes)),
                    name);
            } else if (to_bits < from_bits) {
                PRINT_HIGH("truncate");
                ret = builder.CreateTrunc(
                    shfl, VectorType::get(ret_type, getElementCount(num_lanes)),
                    name);
            } else if (is_unsigned) {
                PRINT_HIGH("zero extend");
                ret = builder.CreateZExt(
                    shfl, VectorType::get(ret_type, getElementCount(num_lanes)),
                    name);
            } else {
                PRINT_HIGH("sign extend");
                ret = builder.CreateSExt(
                    shfl, VectorType::get(ret_type, getElementCount(num_lanes)),
                    name);
            }

            value_cache.setToBeDeleted(inst);
            assert(ret);
            return ret;
        }
        case FunctionResolver::PsimApiEnum::UMULH: {
            llvm::Intrinsic::ID id =
                vf_info.vm_info.function_resolver.Avx512InstrinsicMap[api_enum];
            Function* intrinsic =
                Intrinsic::getDeclaration(inst->getModule(), id);

            // instrinsic works on 32 elements, maybe have this on a table
            int nelem = 32;
            Type* vty = VectorType::get(inst->getType(), nelem, false);
            std::vector<Value*> part_res;
            // for now operate on vector value (even if operand was uniform)
            Value* a = value_cache.getVectorValue(inst->getOperand(0));
            Value* b = value_cache.getVectorValue(inst->getOperand(1));

            for (uint32_t j = 0; j < num_lanes; j += nelem) {
                Value* idx = ConstantInt::get(i64, j);
                Value* sa = builder.CreateExtractVector(vty, a, idx, name);
                Value* sb = builder.CreateExtractVector(vty, b, idx, name);

                SmallVector<Value*> args;
                args.push_back(sa);
                args.push_back(sb);
                Value* sc = builder.CreateCall(intrinsic, args, name);
                part_res.push_back(sc);
            }

            Type* ret_ty = VectorType::get(inst->getType(), num_lanes, false);
            Value* ret = UndefValue::get(ret_ty);
            int j = 0;
            for (auto pr : part_res) {
                Value* idx = ConstantInt::get(i64, j * nelem);
                ret = builder.CreateInsertVector(ret_ty, ret, pr, idx, name);
                j++;
            }
            value_cache.setToBeDeleted(inst);
            return ret;
        }
        case FunctionResolver::PsimApiEnum::COLLECTIVE_ADD_ABS_DIFF: {
            if (inst->getOperand(1)->getType()->getScalarType() !=
                builder.getInt8Ty()) {
                FATAL("Can't transform " << *inst);
            }
            name = "csad.";

            llvm::Intrinsic::ID id =
                vf_info.vm_info.function_resolver.Avx512InstrinsicMap[api_enum];
            Function* intrinsic =
                Intrinsic::getDeclaration(inst->getModule(), id);

            Type* IntrinsicOpType = intrinsic->getArg(0)->getType();
            PointerType* ptr_ty =
                dyn_cast<PointerType>(inst->getOperand(0)->getType());
            assert(ptr_ty);
            Type* ty = ptr_ty->getNonOpaquePointerElementType();
            assert(ty);
            Value* gep = builder.CreateGEP(
                ty, inst->getOperand(0),
                {builder.getInt32(0), builder.getInt32(0)}, name);
            Type* vty = intrinsic->getReturnType();
            Value* acc = builder.CreateLoad(vty, gep, name);

            uint32_t nelem = 64;

            Value* a = value_cache.getVectorValue(inst->getOperand(1));
            Value* b = value_cache.getVectorValue(inst->getOperand(2));

            // zero-off the values of the inactive lanes
            Value* mask = value_cache.getVectorValue(
                vf_info.bb_masks[inst->getParent()].active_mask);
            Value* zero = Constant::getNullValue(a->getType());
            a = builder.CreateSelect(mask, a, zero, name);
            b = builder.CreateSelect(mask, b, zero, name);

            for (uint32_t j = 0; j < num_lanes; j += nelem) {
                Value* idx = ConstantInt::get(i64, j);
                int actual_nelem = std::min(nelem, num_lanes - j);
                Type* eTy =
                    VectorType::get(builder.getInt8Ty(), actual_nelem, false);
                Value* sa = builder.CreateExtractVector(eTy, a, idx, name);
                Value* sb = builder.CreateExtractVector(eTy, b, idx, name);

                Value* sae =
                    builder.CreateVectorSplat(nelem, builder.getInt8(0), name);
                sae = builder.CreateInsertVector(IntrinsicOpType, sae, sa,
                                                 builder.getInt64(0), name);

                Value* sbe =
                    builder.CreateVectorSplat(nelem, builder.getInt8(0), name);
                sbe = builder.CreateInsertVector(IntrinsicOpType, sbe, sb,
                                                 builder.getInt64(0), name);
                SmallVector<Value*> args;
                args.push_back(sae);
                args.push_back(sbe);
                Value* sc = builder.CreateCall(intrinsic, args, name);
                acc = builder.CreateAdd(acc, sc, name);
            }

            builder.CreateStore(acc, gep, false);
            value_cache.setToBeDeleted(inst);
            return inst;
        }

        case FunctionResolver::PsimApiEnum::GANG_SYNC: {
            value_cache.setToBeDeleted(inst);
            return inst;
        }

        case FunctionResolver::PsimApiEnum::ATOMICADD_LOCAL: {
            Value* in = inst->getOperand(0);

            if (!in->getType()->isPointerTy()) {
                FATAL("Can't transform " << *inst);
            }
            PointerType* ptr_ty = dyn_cast<PointerType>(in->getType());
            Type* ty = ptr_ty->getNonOpaquePointerElementType();
            assert(ty);
            Value* loadInst = builder.CreateLoad(ty, in);

            Value* a = value_cache.getVectorValue(inst->getOperand(1));
            Value* mask = value_cache.getVectorValue(
                vf_info.bb_masks[inst->getParent()].active_mask);
            Value* zero = Constant::getNullValue(a->getType());
            Value* selectInst = builder.CreateSelect(mask, a, zero);
            Value* addReduceInst;
            if (ty->isFloatingPointTy()) {
                FastMathFlags fmf;
                fmf.setAllowReassoc(true);
                builder.setFastMathFlags(fmf);
                addReduceInst = builder.CreateFAddReduce(loadInst, selectInst);
                builder.clearFastMathFlags();
            } else if (ty->isIntegerTy()) {
                addReduceInst = builder.CreateAddReduce(selectInst);
                addReduceInst = builder.CreateAdd(loadInst, addReduceInst);
            } else {
                FATAL("Can't transform " << *inst);
            }

            builder.CreateStore(addReduceInst, in);
            value_cache.setToBeDeleted(inst);
            return inst;
        }

        default:
            FATAL("dont' know how to transform " << *inst);
            break;
    }
    return nullptr;
}

Value* TransformStep::transformCallVmath(llvm::CallInst* inst) {
    Function* f = inst->getCalledFunction();
    std::string name = f->getName().str();
    PRINT_HIGH("original math function " << *f);
    // clang-format off
    const std::unordered_map<std::string, std::string> sleef_func_map = {
        {"expf", "Sleef_expf#_u10"},
        {"exp", "Sleef_expd#_u10"},
        {"cos", "Sleef_cosd#_u10"},
        {"cosf", "Sleef_cosf#_u10"},
        {"sin", "Sleef_sind#_u10"},
        {"sinf", "Sleef_sinf#_u10"},
        {"sqrtf", "Sleef_sqrtf#"},
        {"sqrt", "Sleef_sqrtd#"},
        {"logf", "Sleef_logf#_u10"},
        {"logd", "Sleef_logd#_u10"},
        {"powf", "Sleef_powf#_u10"},
        {"pow", "Sleef_powd#_u10"},
        {"fabsf", "Sleef_fabsf#"},
        {"fabs", "Sleef_fabsd#"},
        {"fmax", "Sleef_fmaxd#"},
        {"fmaxf", "Sleef_fmaxf#"},

    };
    // clang-format on

    std::string sleef_func_name = "";
    if (sleef_func_map.find(name) != sleef_func_map.end()) {
        sleef_func_name = sleef_func_map.at(name);
    } else {
        return nullptr;
    }

    // TODO  this should come from the target architecture
    int max_bit_width = 512;
    Type* scalar_ret_ty = f->getFunctionType()->getReturnType();
    TypeSize scalar_ret_ty_size =
        vf_info.data_layout.getTypeAllocSize(scalar_ret_ty->getScalarType());
    int fixed_size = scalar_ret_ty_size.getFixedSize();
    unsigned nelem = max_bit_width / (fixed_size * 8);

    /* replace # with "nelem" */
    size_t index = sleef_func_name.find("#");
    assert(index != std::string::npos);
    std::string l = std::to_string(nelem);
    sleef_func_name.replace(index, 1, l.c_str());

    SmallVector<Type*> vec_argsTy;
    for (uint32_t i = 0; i < f->getFunctionType()->getNumParams(); i++) {
        Type* scalar_arg_ty = f->getFunctionType()->getParamType(i);
        Type* vec_arg_ty =
            VectorType::get(scalar_arg_ty, getElementCount(nelem));
        vec_argsTy.push_back(vec_arg_ty);
    }

    Type* vec_ret_ty = VectorType::get(scalar_ret_ty, getElementCount(nelem));
    Module* mod = inst->getModule();
    FunctionCallee sleef_func;
    if (f->getFunctionType()->getNumParams() == 1) {
        sleef_func = mod->getOrInsertFunction(sleef_func_name.c_str(),
                                              vec_ret_ty, vec_argsTy[0]);
    } else if (f->getFunctionType()->getNumParams() == 2) {
        sleef_func = mod->getOrInsertFunction(
            sleef_func_name.c_str(), vec_ret_ty, vec_argsTy[0], vec_argsTy[1]);
    } else {
        sleef_func = NULL;
        PRINT_HIGH("VMath call not supported "
                   << *inst << " Num params: "
                   << f->getFunctionType()->getNumParams() << "\n");
        return nullptr;
    }

    PRINT_HIGH("transformed math function " << *sleef_func.getCallee());

    Value* a = value_cache.getVectorValue(inst->getOperand(0));
    Value* b = nullptr;
    if (f->getFunctionType()->getNumParams() == 2)
        b = value_cache.getVectorValue(inst->getOperand(1));

    IRBuilder<> builder(inst->getParent());
    builder.SetInsertPoint(inst->getNextNode());

    Type* vty_a = VectorType::get(
        inst->getOperand(0)->getType()->getScalarType(), nelem, false);
    Type* vty_b = nullptr;
    if (f->getFunctionType()->getNumParams() == 2)
        vty_b = VectorType::get(inst->getOperand(1)->getType()->getScalarType(),
                                nelem, false);

    Type* i64 = Type::getInt64Ty(inst->getContext());
    Type* ret_ty = VectorType::get(scalar_ret_ty, getElementCount(num_lanes));
    std::vector<Value*> part_res;
    for (uint32_t j = 0; j < num_lanes; j += nelem) {
        Value* idx = ConstantInt::get(i64, j);
        Value* sa = builder.CreateExtractVector(vty_a, a, idx, name);
        SmallVector<Value*> args;
        args.push_back(sa);
        if (b) {
            Value* sb = builder.CreateExtractVector(vty_b, b, idx, name);
            args.push_back(sb);
        }

        Value* sc = builder.CreateCall(sleef_func, args, name);
        part_res.push_back(sc);
    }

    Value* ret = UndefValue::get(ret_ty);
    int j = 0;
    for (auto pr : part_res) {
        Value* idx = ConstantInt::get(i64, j * nelem);
        ret = builder.CreateInsertVector(ret_ty, ret, pr, idx, name);
        j++;
    }
    value_cache.setToBeDeleted(inst);
    return ret;
}

Value* TransformStep::transformCallIntrinsic(llvm::CallInst* inst) {
    Function* f = inst->getCalledFunction();
    if (!f->isIntrinsic()) {
        return nullptr;
    }

    Intrinsic::ID intrinsic_id = f->getIntrinsicID();

    if (intrinsic_id == Intrinsic::lifetime_start ||
        intrinsic_id == Intrinsic::lifetime_end ||
        intrinsic_id == Intrinsic::dbg_declare ||
        intrinsic_id == Intrinsic::var_annotation ||
        intrinsic_id == Intrinsic::dbg_value) {
        value_cache.setToBeDeleted(inst);
        return inst;
    }

    if (intrinsic_id == Intrinsic::memcpy ||
        intrinsic_id == Intrinsic::memset) {
        // TODO: elision/merging
        return vectorizeUniformCall(inst);
    }

    Type* ty = f->getFunctionType()->getReturnType();
    if (value_cache.getShape(inst).isVarying()) {
        ty = VectorType::get(ty, getElementCount(num_lanes));
    }
    Function* intrinsic_VF =
        Intrinsic::getDeclaration(inst->getModule(), intrinsic_id, {ty});

    if (!intrinsic_VF) {
        return vectorizeUniformCall(inst);
    }
    SmallVector<Value*> args = generateArgsForIntrinsics(inst);
    IRBuilder<> builder(inst->getParent());
    builder.SetInsertPoint(inst->getNextNode());
    Value* ret = builder.CreateCall(intrinsic_VF, args);
    value_cache.setToBeDeleted(inst);
    return ret;
}

SmallVector<Value*> TransformStep::generateArgsForIntrinsics(CallInst* inst) {
    SmallVector<Value*> args;
    for (Use& arg : inst->args()) {
        if (arg->getType()->isMetadataTy()) {
            continue;
        }
        if (value_cache.getShape(inst).isVarying()) {
            args.push_back(value_cache.getVectorValue(arg));
        } else {
            args.push_back(value_cache.getScalarValue(arg));
        }
    }
    return args;
}

Value* TransformStep::transformCallVectFunction(CallInst* inst) {
    Function* f = inst->getCalledFunction();
    VFABI desired_vfabi;

    Type* i1 = Type::getInt1Ty(inst->getContext());
    desired_vfabi.isa = vf_info.vfabi.isa;
    desired_vfabi.mask = vf_info.bb_masks[inst->getParent()].active_mask !=
                         ConstantInt::get(i1, 1);
    desired_vfabi.vlen = vf_info.vfabi.vlen;
    desired_vfabi.scalar_name = f->getName();
    for (Use& arg : inst->args()) {
        if (arg->getType()->isMetadataTy()) {
            PRINT_HIGH("Ignoring metadata argument " << *arg);
            continue;
        }

        /* FIXME alignment */
        Shape shape = value_cache.getShape(arg);
        if (shape.isVarying()) {
            desired_vfabi.parameters.push_back(VFABIShape::Varying());
        } else if (shape.isStrided()) {
            desired_vfabi.parameters.push_back(
                VFABIShape::Strided(shape.getStride()));
        } else {
            desired_vfabi.parameters.push_back(VFABIShape::Uniform());
        }
    }
    desired_vfabi.mangled_name = desired_vfabi.toString();

    FunctionResolution resolution =
        vf_info.vm_info.function_resolver.get(f, desired_vfabi);
    if (!resolution.function) {
        return nullptr;
    }
    PRINT_HIGH("Resolution is " << resolution.function->getName());

    // Call directly, possibly adjusting parameters, lanes, etc.
    VFABI& result_vfabi = resolution.vfabi;
    assert(result_vfabi.isa == desired_vfabi.isa);
    bool all_uniform = true;
    for (auto& i : result_vfabi.parameters) {
        if (i.is_varying || i.stride != 0) {
            all_uniform = false;
            break;
        }
    }
    assert(all_uniform || (result_vfabi.mask == desired_vfabi.mask));
    assert(result_vfabi.vlen == desired_vfabi.vlen);
    std::vector<Value*> args;
    std::vector<Type*> arg_types;

    size_t i = 0;
    for (auto& arg : inst->args()) {
        if (!desired_vfabi.parameters[i].is_varying &&
            result_vfabi.parameters[i].is_varying) {
            args.push_back(value_cache.getVectorValue(arg));
            arg_types.push_back(vf_info.vectorizeType(arg->getType()));
        } else if (desired_vfabi.parameters[i].is_varying) {
            args.push_back(value_cache.getVectorValue(arg));
            arg_types.push_back(vf_info.vectorizeType(arg->getType()));
        } else {
            args.push_back(value_cache.getScalarValue(arg));
            arg_types.push_back(arg->getType());
        }
        i++;
    }

    if (result_vfabi.mask) {
        args.push_back(value_cache.getVectorValue(
            vf_info.bb_masks[inst->getParent()].active_mask));
        arg_types.push_back(vectorizeType(i1, result_vfabi.vlen));
    }

    Type* ret_type;
    if (result_vfabi.return_shape.is_varying) {
        ret_type = vf_info.vectorizeType(inst->getType());
    } else {
        ret_type = inst->getType();
    }
    FunctionType* FT = FunctionType::get(ret_type, arg_types, false);
    CallInst* new_call =
        CallInst::Create(FT, resolution.function, args, inst->getName(), inst);
    new_call->setCallingConv(inst->getCallingConv());
    if (inst->getDebugLoc()) {
        new_call->setDebugLoc(inst->getDebugLoc());
    }
    return new_call;
}

Value* TransformStep::transformPHIFirstPass(PHINode* inst) {
    assert(inst->getNumIncomingValues() > 0);

    /* There are three cases:
     * 1. only one incoming edge
     * 2. one forward edge and one backward edge
     * 3. two forward edges
     */

    // Case 1:
    if (inst->getNumIncomingValues() == 1) {
        PRINT_HIGH("Case 1: one incoming edge");
        return transformSimpleInstruction(inst);
    }

    // Case 2:
    if (vf_info.getPHIBackedge(inst)) {
        PRINT_HIGH("Case 2: backedge");
        // For now, only mutate the type, but do not recursively
        // transform the operands, because that would form a cycle.
        // Instead, fix this in a later phase of transformInstructions()
        if (value_cache.getShape(inst).isVarying()) {
            inst->mutateType(vf_info.vectorizeType(inst->getType()));
        }
        return inst;
    }

    // Case 3: two forward edges
    PRINT_HIGH("Case 3: two forward edges");

    // If the PHI is uniform or strided, we don't have to translate it into a
    // select
    Shape shape = value_cache.getShape(inst);
    if (!shape.isVarying()) {
        return transformInstructionWithoutVectorizing(inst);
    }

    PRINT_HIGH(
        "Varying PHI has two forward edges and is varying; converting to "
        "select");
    assert(inst->getNumIncomingValues() == 2);

    // Get the dominator as well
    BasicBlock* a = inst->getIncomingBlock(0);
    BasicBlock* b = inst->getIncomingBlock(1);
    BasicBlock* dominator = vf_info.getDominator(a, b);

    // Now, create the select
    // To see why it uses the original phi 'inst' as an argument, consider
    // this example:
    //
    // BB0:
    // a = ...
    // br cond BB1, BB2
    //
    // BB1:
    // b = ...
    // br BB2
    //
    // BB2:
    // c = phi([BB0, a], [BB1, b])
    //
    // we want to replace 'c' with:
    // c = select BB1mask, b, a
    //
    // however, LLVM complains because b does not dominate c in that case.
    // that's why the phi 'c' is there to begin with.
    //
    // solution: keep the phi, and use it in place of b in the select.
    // if BB1.any(), then c == b, and the select does what we wanted.
    // else, c2 always picks 'a' anyway.
    // c = phi([BB0, a], [BB1, b])
    // new_final_value = select BB1mask, c, a
    Value* mask = vf_info.getPHISelectMask(inst);

    // First vectorize the PHI itself, just as a simple instruction
    Value* vectorized_phi = transformSimpleInstruction(inst);
    assert(vectorized_phi == inst);

    // Create a vector select that gets added after the PHI
    Instruction* select = SelectInst::Create(
        value_cache.getVectorValue(mask), vectorized_phi,
        value_cache.getVectorValue(inst->getIncomingValueForBlock(dominator)),
        inst->getName() + ".",
        inst->getParent()->getFirstNonPHIOrDbgOrLifetime());
    PRINT_HIGH("Select is " << *select);
    return select;
}

Value* TransformStep::transformPHISecondPass(PHINode* inst) {
    // we already mutated the type in case 2 of the forward pass, so don't
    // re-mutate the type.  Just update the operands

    for (unsigned i = 0; i < inst->getNumOperands(); i++) {
        Value* v;
        if (value_cache.getShape(inst).isVarying()) {
            v = value_cache.getVectorValue(inst->getOperand(i));
        } else {
            v = value_cache.getScalarValue(inst->getOperand(i));
        }

        if (!v) {
            FATAL("No transformed value for " << *inst->getOperand(i) << "\n");
            assert(false);
        }

        inst->setOperand(i, v);
    }

    return inst;
}

Value* TransformStep::transformAlloca(AllocaInst* inst) {
    // TODO: struct layout optimizations
    PRINT_HIGH("Original alloca instruction is " << *inst);

    if (inst->getAllocatedType()->isStructTy()) {
        vf_info.diagnostics.unoptimized_allocas.push_back(valueString(inst));
    }

    if (value_cache.getArrayLayoutOpt(inst)) {
        return transformInstructionWithoutVectorizing(inst);
    }
    // Allocate 'num_lanes' elements
    ConstantInt* orig_num_elements = cast<ConstantInt>(inst->getArraySize());
    APInt new_num_elements_int = orig_num_elements->getValue() * num_lanes;
    Constant* new_num_elements =
        ConstantInt::get(orig_num_elements->getType(), new_num_elements_int);

    // Always allocate an i8 array of size properly padded to make sure that
    // each individual allocation can be properly aligned
    Type* ty = inst->getAllocatedType();
    TypeSize type_size = vf_info.data_layout.getTypeAllocSize(ty);
    uint64_t layout_stride = type_size.getFixedSize();
    uint64_t align = inst->getAlign().value();
    uint64_t padded_size = roundUp(layout_stride, align);
    Type* i8 = Type::getInt8Ty(inst->getContext());
    Type* padded_array_type = ArrayType::get(i8, padded_size);

    Instruction* new_alloca =
        new AllocaInst(padded_array_type, 0, new_num_elements, inst->getAlign(),
                       inst->getName() + ".");
    new_alloca->insertAfter(inst);

    PRINT_HIGH("New alloca is " << *new_alloca);

    // Get pointers to the base address of each individual allocation
    Instruction* gep = GetElementPtrInst::Create(
        padded_array_type, new_alloca,
        {vf_info.getLaneID(orig_num_elements->getValue().getSExtValue())},
        inst->getName() + ".");
    gep->insertAfter(new_alloca);
    PRINT_HIGH("New GEP is " << *gep);

    // Cast back to the originally expected type
    Type* result_ty =
        VectorType::get(inst->getType(), getElementCount(num_lanes));
    Instruction* cast = new BitCastInst(gep, result_ty, inst->getName() + ".");
    cast->insertAfter(gep);
    PRINT_HIGH("New BitCast is " << *cast);

    value_cache.setToBeDeleted(inst);
    return cast;
}

Value* TransformStep::transformInstructionWithoutVectorizing(
    Instruction* inst) {
    PRINT_HIGH("Transforming instruction without vectorizing: " << *inst);

    for (unsigned i = 0; i < inst->getNumOperands(); i++) {
        Value* v;
        if (inst->getOperand(i)->getType()->isVectorTy()) {
            v = value_cache.getVectorValue(inst->getOperand(i));
        } else {
            v = value_cache.getScalarValue(inst->getOperand(i));
        }
        inst->setOperand(i, v);
    }

    return inst;
}

Value* TransformStep::transformExtractInsertElement(Instruction* inst,
                                                    bool isExtract) {
    std::string s = isExtract ? "ExtractElementInst " : "InsertElementInst ";
    int ret_op = isExtract ? 0 : 1;
    int in_op = isExtract ? 1 : 2;

    CallInst* call = dyn_cast<CallInst>(inst->getOperand(in_op));
    if (!call || (vf_info.vm_info.function_resolver.getPsimApiEnum(
                      call->getCalledFunction()) !=
                  FunctionResolver::PsimApiEnum::GET_LANE_NUM)) {
        FATAL(s << *inst << "; does not use psim_get_lane_num() instead "
                << *inst->getOperand(in_op));
    }

    value_cache.setToBeDeleted(inst);
    return value_cache.getVectorValue(inst->getOperand(ret_op));
}

Value* TransformStep::transformInstruction(Instruction* inst) {
    PRINT_MID("");
    PRINT_MID("Transforming instruction "
              << *inst << " ; " << value_cache.getShape(inst).toString());

    // the "simple" case: just transform all the operands
    if (isa<UnaryOperator>(inst) || isa<BinaryOperator>(inst) ||
        isa<CastInst>(inst) || isa<CmpInst>(inst) ||
        isa<GetElementPtrInst>(inst) || isa<SelectInst>(inst) ||
        isa<SIToFPInst>(inst) || isa<FreezeInst>(inst)) {
        return transformSimpleInstruction(inst);
    }

    // AllocaInst
    AllocaInst* alloca = dyn_cast<AllocaInst>(inst);
    if (alloca) {
        return transformAlloca(alloca);
    }

    // LoadInst or StoreInst
    LoadInst* load = dyn_cast<LoadInst>(inst);
    StoreInst* store = dyn_cast<StoreInst>(inst);
    if (load || store) {
        return transformMemInst(inst);
    }

    // BranchInst
    BranchInst* br = dyn_cast<BranchInst>(inst);
    if (br) {
        return transformBranch(br);
    }

    // CallInst
    CallInst* call = dyn_cast<CallInst>(inst);
    if (call) {
        return transformCall(call);
    }

    // PHINode
    PHINode* phi = dyn_cast<PHINode>(inst);
    if (phi) {
        return transformPHIFirstPass(phi);
    }

    // ReturnInst
    ReturnInst* ret = dyn_cast<ReturnInst>(inst);
    if (ret) {
        return transformReturn(ret);
    }

    // ExtracElementtInst
    ExtractElementInst* extract = dyn_cast<ExtractElementInst>(inst);
    if (extract) {
        return transformExtractInsertElement(inst, true);
    }

    // InsertElementInst
    InsertElementInst* insert = dyn_cast<InsertElementInst>(inst);
    if (insert) {
        return transformExtractInsertElement(inst, false);
    }

    FATAL("Don't know how to transform instruction '"
          << *inst << "' of type '" << *inst->getType() << "' with shape "
          << value_cache.getShape(inst).toString() << " !\n");
}

void TransformStep::transform() {
    populateDisplayWarnings();

    PRINT_LOW("Transforming instructions:");

    // Iterate over the instructions and transform them
    for (Instruction* I : vf_info.instruction_order) {
        Value* v = transformInstruction(I);

        assert(value_cache.has(I));
        if (value_cache.getShape(I).isVarying()) {
            value_cache.setVectorValue(I, v);
        } else {
            value_cache.setScalarValue(I, v);
        }
    }

    // Vectorize the backedge PHIs that we skipped earlier
    PRINT_MID("\nSecond pass: vectorize PHIs with backedges");
    for (Instruction* I : vf_info.instruction_order) {
        PHINode* phi = dyn_cast<PHINode>(I);
        if (!phi || !vf_info.getPHIBackedge(phi)) {
            continue;
        }
        PRINT_MID("\nVectorizing backedge PHI " << *phi);
        transformPHISecondPass(phi);
    }

    // Delete any now-obsolete instructions
    value_cache.deleteObsoletedInsts();
}

void TransformStep::populateDisplayWarnings() {
    /* order instructions in line order */
    std::map<std::string, Instruction*> line_ordered;
    int cnt = 0;
    for (BasicBlock& BB : *vf_info.VF) {
        for (Instruction& I : BB) {
            int leading_zeros = 6;
            std::string str = getDebugLocStr(&I, leading_zeros);
            if (line_ordered.find(str) != line_ordered.end()) {
                str += "." + std::to_string(cnt++);
            }
            line_ordered[str] = &I;
        }
    }

    bool is_warning_on = true;

    /* iterate on instructions */
    for (auto it : line_ordered) {
        Instruction* I = it.second;
        // PRINT_ALWAYS(it.first << " " << *I << " - warning on? " <<
        // is_warning_on);
        if (is_warning_on) {
            display_warnings.insert(I);
        }
        if (global_opts.ignore_warn_set) {
            continue;
        }

        CallInst* call = dyn_cast<CallInst>(I);
        if (!call) {
            continue;
        }
        Function* f = call->getCalledFunction();
        if (!f || !f->isIntrinsic() ||
            f->getIntrinsicID() != Intrinsic::var_annotation) {
            continue;
        }
        ConstantExpr* ce = cast<ConstantExpr>(call->getOperand(1));
        if (!ce || ce->getOpcode() != Instruction::GetElementPtr) {
            continue;
        }
        if (GlobalVariable* annoteStr =
                dyn_cast<GlobalVariable>(ce->getOperand(0))) {
            if (ConstantDataSequential* data = dyn_cast<ConstantDataSequential>(
                    annoteStr->getInitializer())) {
                if (data->isString()) {
                    std::string str = data->getAsString().str();
                    if (str.find("warn_on") != std::string::npos) {
                        is_warning_on = true;
                    } else if (str.find("warn_off") != std::string::npos) {
                        is_warning_on = false;
                    }
                }
            }
        }
    }
}

std::unordered_set<std::string> TransformStep::already_warned;

void TransformStep::printWarning(Instruction* inst, std::string msg) {
    std::string loc_str = getDebugLocStr(inst);
    if (display_warnings.find(inst) != display_warnings.end() &&
        already_warned.find(loc_str) == already_warned.end()) {
        already_warned.insert(loc_str);
        WARNING(loc_str + " " + msg);
        if (global_opts.error_on_warn) {
            errs() << "Fatal error: Error on warning enabled!\n";
            exit(1);
        }
    }
}

}  // namespace ps
