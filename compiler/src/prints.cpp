/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "prints.h"

#include <cstdarg>

using namespace llvm;

namespace ps {

unsigned prints_verbosity_level;
[[maybe_unused]] static unsigned& verbosity_level = prints_verbosity_level;

AddPrintsStep::AddPrintsStep(VectorizedFunctionInfo& vf_info)
    : vf_info(vf_info), builder(vf_info.mod->getContext()) {}

Value* AddPrintsStep::getGlobalString(std::string s) {
    auto i = global_strings.find(s);
    if (i != global_strings.end()) {
        return i->second;
    }

    Value* v =
        builder.CreateGlobalStringPtr(s, "global_string", 0, vf_info.mod);
    global_strings[s] = v;
    return v;
}

Instruction* AddPrintsStep::addPrintf(Instruction* inst, const char* format,
                                      ...) {
    Function* func_printf = vf_info.mod->getFunction("printf");
    if (!func_printf) {
        PRINT_HIGH("Did not find printf!\n");
        FunctionType* FuncTy = FunctionType::get(
            IntegerType::get(vf_info.mod->getContext(), 32), true);

        func_printf = Function::Create(FuncTy, GlobalValue::ExternalLinkage,
                                       "printf", vf_info.mod);
        func_printf->setCallingConv(CallingConv::C);
    }

    Value* str = getGlobalString(std::string(format));
    std::vector<Value*> call_params;
    call_params.push_back(str);

    va_list ap;
    va_start(ap, format);

    while (1) {
        llvm::Value* op = va_arg(ap, llvm::Value*);
        if (op) {
            call_params.push_back(op);
        } else {
            break;
        }
    }
    va_end(ap);

    return CallInst::Create(func_printf, call_params, "call", inst);
}

void AddPrintsStep::addValuePrint(Value* V, Instruction* term) {
    Type* ty = V->getType();
    Type* scalar_ty = ty->getScalarType();
    unsigned bits = scalar_ty->getScalarSizeInBits();
    PRINT_HIGH("Value scalar type " << *scalar_ty << " has " << bits
                                    << " bits\n");
    assert(bits <= 64);

    unsigned print_bits;
    std::string fmt;
    if (isa<PointerType>(scalar_ty)) {
        print_bits = 0;
        fmt = "0x%016lx ";
    } else if (bits == 0) {
        return;
    } else if (bits == 1) {
        print_bits = 32;
        fmt = "%d";
    } else if (bits <= 32) {
        print_bits = 32;
        fmt = "0x%016x ";
    } else if (bits <= 64) {
        print_bits = 64;
        fmt = "0x%016lx ";
    } else {
        assert(false);
    }
    assert(bits <= print_bits);

    addPrintf(term, "    %030s: ", getGlobalString(V->getName().str()),
              nullptr);

    Type* i32 = Type::getInt32Ty(vf_info.ctx);

    unsigned num_lanes = ty->isVectorTy() ? vf_info.num_lanes : 1;
    for (unsigned i = 0; i < num_lanes; i++) {
        Value* extracted;
        if (ty->isVectorTy()) {
            extracted = ExtractElementInst::Create(
                V, ConstantInt::get(i32, i),
                V->getName() + "_extract" + std::to_string(i), term);
        } else {
            extracted = V;
        }

        Value* print_value;
        if (isa<PointerType>(scalar_ty) || bits == print_bits) {
            print_value = extracted;
        } else {
            Type* ext_type = Type::getIntNTy(vf_info.ctx, print_bits);
            print_value = new ZExtInst(
                extracted, ext_type,
                V->getName() + "_extend" + std::to_string(i), term);
        }

        addPrintf(term, fmt.c_str(), print_value, nullptr);
    }
    addPrintf(term, "\n", nullptr);
}

void AddPrintsStep::addPrints() {
    for (BasicBlock& BB : *vf_info.VF) {
        Instruction* term = BB.getTerminator();
        Value* bb_name = getGlobalString(BB.getName().str());
        addPrintf(term, "Basic block %s:\n", bb_name, nullptr);

        // Stop at the first inserted printf, so we don't recurse.  The
        // printf will be at the spot currently occupied by the terminator
        size_t stop = BB.size() - 2;
        size_t count = 0;
        for (Instruction& I : BB) {
            if (count == stop) {
                PRINT_HIGH("Stopping at " << I << "\n");
                break;
            }
            count++;

            PRINT_HIGH("Adding printf for instruction " << I << "\n");

            Value* I_str_value = getGlobalString(valueString(&I));

            // Print the instruction opcode
            addPrintf(term, "  %s\n", I_str_value, nullptr);

            // Print the instruction operand input values
            for (auto& op : I.operands()) {
                // LLVM does not allow taking the address of intrinsics
                Function* F = dyn_cast<Function>(op);
                if (F && F->isIntrinsic()) {
                    continue;
                }

                addValuePrint(op, term);
            }

            // Print the instruction output, if applicable
            addValuePrint(&I, term);

            // Newline separating instructions
            addPrintf(term, "\n", nullptr);
        }
        addPrintf(term, "\n", nullptr);
    }
}

}  // namespace ps
