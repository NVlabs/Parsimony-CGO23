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

#include <string>
#include <unordered_map>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>

#include "vectorize.h"

namespace ps {

extern unsigned prints_verbosity_level;

class AddPrintsStep {
  public:
    AddPrintsStep(VectorizedFunctionInfo& vf_info);
    void addPrints();

  private:
    VectorizedFunctionInfo& vf_info;
    llvm::IRBuilder<> builder;
    std::unordered_map<std::string, llvm::Value*> global_strings;

    llvm::Instruction* addPrintf(llvm::Instruction* inst, const char* format,
                                 ...);
    llvm::Value* getGlobalString(std::string s);
    void addValuePrint(llvm::Value* V, llvm::Instruction* term);
};

}  // namespace ps
