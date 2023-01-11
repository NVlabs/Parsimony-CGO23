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

extern unsigned inst_order_verbosity_level;

class InstructionOrderStep {
  public:
    InstructionOrderStep(ps::VectorizedFunctionInfo& vf_info)
        : vf_info(vf_info), instruction_order(vf_info.instruction_order) {}
    void calculate();

  private:
    ps::VectorizedFunctionInfo& vf_info;
    std::vector<llvm::Instruction*>& instruction_order;

    bool OperandDominatedByUser(llvm::Instruction* operand,
                                llvm::Instruction* user);
};

}  // namespace ps
