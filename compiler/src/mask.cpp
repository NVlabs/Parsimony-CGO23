/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "mask.h"

#include <cassert>
#include <cstdarg>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <llvm/Analysis/VectorUtils.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils.h>

#include "vectorize.h"

using namespace llvm;

namespace ps {

unsigned mask_verbosity_level;
[[maybe_unused]] static unsigned& verbosity_level = mask_verbosity_level;

MasksStep::MasksStep(VectorizedFunctionInfo& vf_info) : vf_info(vf_info) {}

Value* MasksStep::calculateEntryMaskFromPredecessor(BasicBlock* BB) {
    Loop* loop = vf_info.loop_info->getLoopFor(BB);
    if (!loop || BB != loop->getExitingBlock()) {
        // simple straight-line cases
        return vf_info.bb_masks[BB].active_mask;
    }

    // active_mask would hold only the active mask for the last iteration of
    // the loop, so we want the entry_mask here: the set of threads that
    // entered the loop in the beginning
    return vf_info.bb_masks[loop->getHeader()].entry_mask;
}

void MasksStep::calculateBBMaskEntry(BasicBlock* BB) {
    Type* i1 = Type::getInt1Ty(vf_info.ctx);

    Value* function_entry_mask;
    if (vf_info.vfabi.mask) {
        Function* F = BB->getParent();
        function_entry_mask = F->getArg(F->arg_size() - 1);
    } else {
        function_entry_mask = ConstantInt::get(i1, 1);
    }

    vf_info.bb_masks[BB].entry_mask = function_entry_mask;
    vf_info.bb_masks[BB].active_mask = function_entry_mask;
    PRINT_HIGH("BasicBlock " << BB->getName()
                             << " uses the function entry mask");
}

void MasksStep::calculateBBMaskLoopHeader(BasicBlock* BB) {
    // We don't think we should hit this given we're in structurized
    // form, but if we do, deal with it then
    if (!BB->hasNPredecessors(2)) {
        FATAL("BasicBlock " << BB->getName()
                            << " does not have 2 predecessors");
    }

    // Get the two predecessors
    auto i = pred_begin(BB);
    BasicBlock* a = *i;
    i++;
    BasicBlock* b = *i;

    // Assert that one is a preheader incoming edge, and the other is a
    // backedge
    BasicBlock *preheader, *loopback;
    if (vf_info.doms->dominates(a, BB) && vf_info.doms->dominates(BB, b)) {
        preheader = a;
        loopback = b;
    } else if (vf_info.doms->dominates(b, BB) &&
               vf_info.doms->dominates(BB, a)) {
        preheader = b;
        loopback = a;
    } else {
        FATAL("Unexpected predecessor pattern for " << BB->getName());
    }

    // Recurse, in case we haven't gotten to a or b yet
    calculateBBMasks(preheader);
    // don't skip forward and calculateBBMasks(loopback)

    vf_info.bb_masks[BB].entry_mask =
        calculateEntryMaskFromPredecessor(preheader);

    Type* i1 = Type::getInt1Ty(vf_info.ctx);
    PHINode* phi = PHINode::Create(i1, 2, BB->getName() + "_loop_active_mask");
    phi->addIncoming(vf_info.bb_masks[BB].entry_mask, preheader);
    phi->setIncomingBlock(1, loopback);
    // set the loopback condition later, once we've reached it
    loop_header_active_mask_phis.push_back(phi);
    phi->insertBefore(&BB->front());

    vf_info.bb_masks[BB].active_mask = phi;

    PRINT_HIGH("BasicBlock " << BB->getName() << " is a loop header");
    PRINT_HIGH("  entry mask is: " << *vf_info.bb_masks[BB].entry_mask);
    PRINT_HIGH("  active mask is: " << *phi);
}

void MasksStep::calculateBBMaskSinglePredecessor(BasicBlock* BB) {
    BasicBlock* predecessor = BB->getSinglePredecessor();

    // Recurse, in case we haven't gotten to predecessor yet
    calculateBBMasks(predecessor);

    // If the structurizer pass somehow ever leaves any terminator other
    // than a BranchInst, deal with it then
    BranchInst* term = cast<BranchInst>(predecessor->getTerminator());

    // There are three cases:
    // 1) unconditional branch
    // 2) predecessor is loop tail
    // 3) predecessor is 'if', BB is 'then' in an if-then pattern

    // For cases 1 and 2, calculate the entry mask from the predecessor
    Loop* loop = vf_info.loop_info->getLoopFor(predecessor);
    bool predecessor_is_loop_tail =
        loop && predecessor == loop->getExitingBlock();
    if (term->isUnconditional() || predecessor_is_loop_tail) {
        vf_info.bb_masks[BB].entry_mask =
            calculateEntryMaskFromPredecessor(predecessor);
        vf_info.bb_masks[BB].active_mask = vf_info.bb_masks[BB].entry_mask;
        PRINT_HIGH("BasicBlock "
                   << BB->getName()
                   << " inherits masks from its single predecessor "
                   << predecessor->getName());
        return;
    }

    // Otherwise, case 3: this is a 'then' block in an if-then pattern

    // Per structurizer source
    // (lib/Transforms/Scalar/StrucurizeCFG.cpp), the true exit goes to
    // the 'then' block, and the false exit skips the 'then' block
    assert(term->getSuccessor(0) == BB);

    // Create the new BB mask
    Value* predecessor_active_mask =
        calculateEntryMaskFromPredecessor(predecessor);
    Instruction* mask = BinaryOperator::Create(
        BinaryOperator::And, predecessor_active_mask, term->getCondition(),
        BB->getName() + "_active_mask");
    mask->insertBefore(term);

    // Modify 'term' to use the newly created mask
    term->setCondition(mask);

    // Store the results
    vf_info.bb_masks[BB].entry_mask = mask;
    vf_info.bb_masks[BB].active_mask = mask;
    PRINT_HIGH("BasicBlock " << BB->getName()
                             << " creates a mask from single predecessor "
                             << predecessor->getName() << ": " << *mask);
}

void MasksStep::calculateBBMaskTwoPredecessors(BasicBlock* BB) {
    auto i = pred_begin(BB);
    BasicBlock* a = *i;
    i++;
    BasicBlock* b = *i;

    // Recurse, in case we haven't gotten to a or b yet
    calculateBBMasks(a);
    calculateBBMasks(b);

    // Figure out which predecessor is the 'if' block, i.e., which
    // is the dominator of the other
    BasicBlock* dominator = vf_info.getDominator(a, b);

    // Inherit the masks from the dominator
    vf_info.bb_masks[BB] = vf_info.bb_masks[dominator];
    PRINT_HIGH("BasicBlock " << BB->getName()
                             << " inherits a mask from dominator predecessor "
                             << dominator->getName());
}

void MasksStep::calculateBBMasks(BasicBlock* BB) {
    if (vf_info.bb_masks.find(BB) != vf_info.bb_masks.end()) {
        return;
    }

    // Create the entry in 'vf_info.bb_masks' so recursion bottoms out
    PRINT_HIGH("Calculating masks for BasicBlock " << BB->getName());
    vf_info.bb_masks[BB].active_mask = nullptr;

    // Entry block
    if (BB->hasNPredecessors(0)) {
        calculateBBMaskEntry(BB);
        return;
    }

    // Check for loop headers
    if (vf_info.loop_info->isLoopHeader(BB)) {
        calculateBBMaskLoopHeader(BB);
        return;
    }

    // Non-loop, single predecessor
    BasicBlock* predecessor = BB->getSinglePredecessor();
    if (predecessor) {
        calculateBBMaskSinglePredecessor(BB);
        return;
    }

    // Non-loop, two predecessors
    if (BB->hasNPredecessors(2)) {
        calculateBBMaskTwoPredecessors(BB);
        return;
    }

    FATAL("BasicBlock " << BB->getName() << " has more than 2 predecessors?");
}

void MasksStep::finalizeMaskPHIs() {
    for (PHINode* phi : loop_header_active_mask_phis) {
        BasicBlock* BB = phi->getIncomingBlock(1);
        BranchInst* term = cast<BranchInst>(BB->getTerminator());

        // per StructurizeCFG source, the backedge is always the false
        // branch
        assert(term->isConditional());
        Type* i1 = Type::getInt1Ty(vf_info.ctx);
        Instruction* condition_inv = BinaryOperator::Create(
            BinaryOperator::Xor, term->getCondition(), ConstantInt::get(i1, 1),
            BB->getName() + "_repeat_mask", term);

        // Swap the terminator polarity, and use this inverted condition
        // as the new branch condition
        term->setCondition(condition_inv);
        term->swapSuccessors();

        // Update the PHINode itself
        phi->addIncoming(condition_inv, BB);
        assert(phi->getNumOperands() == 2);
    }
}

void MasksStep::calculate() {
    PRINT_MID("\n");
    PRINT_LOW("Calculating basic block masks:");
    PRINT_MID("");

    // Iterate over the basic blocks and calcluate masks
    for (BasicBlock& BB : *vf_info.VF) {
        calculateBBMasks(&BB);
    }

    // Fix up the PHIs that were deferred until the end due to pulling from
    // mask values that haven't yet been calculated (masks that are farther
    // ahead in the CFG somewhere)
    finalizeMaskPHIs();
}

}  // namespace ps
