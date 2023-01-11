/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#pragma once

#include <vector>

#include <llvm/IR/Function.h>

#include "vectorize.h"

namespace ps {

extern unsigned mask_verbosity_level;

class MasksStep {
  public:
    MasksStep(VectorizedFunctionInfo& vf_info);
    void calculate();

  private:
    VectorizedFunctionInfo& vf_info;
    std::vector<llvm::PHINode*> loop_header_active_mask_phis;

    llvm::Value* calculateEntryMaskFromPredecessor(llvm::BasicBlock* BB);
    void finalizeMaskPHIs();

    void calculateBBMaskEntry(llvm::BasicBlock* BB);
    void calculateBBMaskLoopHeader(llvm::BasicBlock* BB);
    void calculateBBMaskSinglePredecessor(llvm::BasicBlock* BB);
    void calculateBBMaskTwoPredecessors(llvm::BasicBlock* BB);
    void calculateBBMasks(llvm::BasicBlock* BB);
};

}  // namespace ps
