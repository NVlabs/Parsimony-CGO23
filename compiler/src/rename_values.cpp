/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "llvm/IR/Function.h"

using namespace llvm;

namespace ps {

void renameValues(Function& F) {
    int bb_count = 0;
    int inst_count = 0;

    int arg_id = 0;
    for (Argument& arg : F.args()) {
        if (!arg.hasName()) {
            arg.setName("arg" + std::to_string(arg_id));
        }
    }

    for (BasicBlock& BB : F) {
        if (!BB.hasName()) {
            std::string name = "BB" + std::to_string(bb_count);
            BB.setName(name);
        }
        bb_count++;

        for (Instruction& I : BB) {
            if (!I.hasName() && !I.getType()->isVoidTy()) {
                std::string name = "INST" + std::to_string(inst_count);
                I.setName(name);
            }
            inst_count++;
        }
    }
}

}  // namespace ps
