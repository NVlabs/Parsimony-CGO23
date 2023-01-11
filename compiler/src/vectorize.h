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

#include <z3++.h>
#include <vector>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Value.h>
#include <llvm/Passes/PassBuilder.h>

#include "broadcast.h"
#include "resolver.h"
#include "shape.h"
#include "utils.h"
#include "value_cache.h"
#include "vfabi.h"

namespace ps {

extern unsigned vectorize_verbosity_level;

struct VectorizedModuleInfo;

struct VectorizedFunctionInfo {
    VectorizedFunctionInfo(VectorizedModuleInfo& vm_info, llvm::Function* VF,
                           VFABI& vfabi);

    // Module info, and local copies of its members (for convenience)
    VectorizedModuleInfo& vm_info;
    llvm::LLVMContext& ctx;
    llvm::Module* mod;

    // Function-specific members
    llvm::Function* VF = nullptr;
    ps::VFABI vfabi;
    int num_lanes;
    ValueCache value_cache;

    // General LLVM analyses
    llvm::FunctionAnalysisManager FAM;
    llvm::PassBuilder PB;
    llvm::FunctionPassManager FPM;
    llvm::DominatorTree* doms;
    llvm::LoopInfo* loop_info;
    llvm::DataLayout data_layout;

    // LLVM analysis step
    void getAnalyses();

    // Mask generation step
    struct BasicBlockInfo {
        llvm::Value* active_mask = nullptr;
        llvm::Value* entry_mask = nullptr;
    };
    std::unordered_map<llvm::BasicBlock*, BasicBlockInfo> bb_masks;

    // Instruction order step
    std::vector<llvm::Instruction*> instruction_order;

    // Verification
    void verifyTransformedFunction();

    // Common helper functions
    llvm::Type* vectorizeType(llvm::Type* ty);
    llvm::BasicBlock* getDominator(llvm::BasicBlock* a, llvm::BasicBlock* b);
    /* Which mask will each PHI node use when converting to a select instruction
     * during vectorization?
     *
     * A
     * |\
     * | B
     * |/
     * C:
     * phi([x, A], [y, B])  <-- will use B active mask as the select mask
     */
    llvm::Value* getPHISelectMask(llvm::PHINode* phi,
                                  bool* is_inverted = nullptr);
    llvm::BasicBlock* getPHIBackedge(llvm::PHINode* inst);
    llvm::Value* getLaneID(int stride = 1);

    // z3 context, for shape analysis
    z3::context z3_ctx;
    z3::solver solver;

    // Diagnostics
    struct Diagnostics {
        std::set<std::string> unhandled_shape_opcodes;
        std::vector<std::string> unhandled_shape_insts;

        std::unordered_map<size_t, std::vector<std::string>> gathers;
        std::unordered_map<size_t, std::vector<std::string>> scatters;

        std::set<std::string> scalarized_called_functions;

        std::vector<std::string> function_pointer_calls;

        std::vector<std::string> unoptimized_allocas;
    } diagnostics;
};

typedef std::unordered_map<llvm::Function*,
                           std::vector<VectorizedFunctionInfo*>>
    VFInfoMap;

struct VectorizedModuleInfo {
    VectorizedModuleInfo(llvm::Module* mod);

    llvm::LLVMContext& ctx;
    llvm::Module* mod;
    VFInfoMap vfinfo_map;
    FunctionResolver function_resolver;
};

}  // namespace ps
