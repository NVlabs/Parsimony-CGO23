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

#include <unordered_set>
#include <vector>

#include <llvm/IR/Function.h>

#include "shape.h"
#include "vectorize.h"

namespace ps {

extern unsigned transform_verbosity_level;

/* Instruction transforms:
 * "Transform": convert an instruction into its final form, whether that
 * means vectorizing or leaving as scalar (but still updating operands as
 * necessary)
 *
 * PHI nodes have some special handling in that they are transformed in two
 * passes.  The first pass iterates from beginning to end in
 * instruction_order, and partly skips PHIs with incoming backedges in order
 * to break any dependency cycles.  After the first pass ends, we do a
 * second_pass which finishes transforming those PHIs.
 */
class TransformStep {
  public:
    TransformStep(VectorizedFunctionInfo& vf_info);
    void transform();

  private:
    VectorizedFunctionInfo& vf_info;
    ValueCache& value_cache;
    unsigned num_lanes;
    std::unordered_set<llvm::Instruction*> display_warnings;
    static std::unordered_set<std::string> already_warned;

    llvm::Value* transformInstruction(llvm::Instruction* inst);
    llvm::Value* transformInstructionWithoutVectorizing(
        llvm::Instruction* inst);
    llvm::Value* transformSimpleInstruction(llvm::Instruction* inst);
    llvm::Value* transformAlloca(llvm::AllocaInst* inst);
    llvm::Value* transformBranch(llvm::BranchInst* inst);

    /* Transform calls group */
    llvm::Value* transformCall(llvm::CallInst* inst);
    llvm::Value* transformCallPsimApi(llvm::CallInst* inst);
    llvm::Value* transformCallIntrinsic(llvm::CallInst* inst);
    llvm::Value* transformCallVmath(llvm::CallInst* inst);
    llvm::Value* transformCallVectFunction(llvm::CallInst* inst);

    llvm::Value* transformLoad(llvm::LoadInst* inst);
    llvm::Value* transformPHIFirstPass(llvm::PHINode* inst);
    llvm::Value* transformPHISecondPass(llvm::PHINode* inst);
    llvm::Value* transformReturn(llvm::ReturnInst* inst);
    llvm::Value* transformMemInst(llvm::Instruction* inst);
    llvm::Value* transformExtractInsertElement(llvm::Instruction* inst,
                                               bool isExtract);
    llvm::Value* vectorizeUniformCall(llvm::CallInst* inst);

    llvm::Value* vectorizeMemInst(llvm::Instruction* inst, bool packed,
                                  std::vector<int> indices = {},
                                  size_t esize = 0);

    llvm::Value* generateMaskForMemInst(llvm::Instruction* inst,
                                        std::vector<int> indices = {},
                                        unsigned factor = 1);
    void rebaseMemPackedIndices(std::vector<int>& indices, int& min_index,
                                unsigned& factor);

    llvm::SmallVector<llvm::Value*> generateArgsForIntrinsics(
        llvm::CallInst* inst);

    void populateDisplayWarnings();
    void printWarning(llvm::Instruction* inst, std::string msg);
};

}  // namespace ps
