/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <cassert>
#include <unordered_map>
#include <vector>

#include <llvm/Analysis/DomTreeUpdater.h>
#include <llvm/Analysis/VectorUtils.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>

#include "utils.h"
#include "vectorize.h"

using namespace llvm;
using namespace ps;

namespace ps {

unsigned vectorize_verbosity_level;
[[maybe_unused]] static unsigned& verbosity_level = vectorize_verbosity_level;

VectorizedFunctionInfo::VectorizedFunctionInfo(VectorizedModuleInfo& vm_info,
                                               Function* VF, VFABI& vfabi)
    : vm_info(vm_info),
      ctx(vm_info.ctx),
      mod(vm_info.mod),
      VF(VF),
      vfabi(vfabi),
      num_lanes(vfabi.vlen),
      value_cache(VF, num_lanes, this),
      data_layout(mod),
      solver(z3_ctx) {}

Type* VectorizedFunctionInfo::vectorizeType(Type* ty) {
    return ps::vectorizeType(ty, num_lanes);
}

Value* VectorizedFunctionInfo::getLaneID(int stride) {
    Type* i32 = Type::getInt32Ty(ctx);
    return createStrideConstant(ConstantInt::get(i32, 0), num_lanes, stride);
}

void VectorizedFunctionInfo::getAnalyses() {
    PB.registerFunctionAnalyses(FAM);
    FPM.run(*VF, FAM);

    loop_info = &FAM.getResult<LoopAnalysis>(*VF);
    doms = &FAM.getResult<DominatorTreeAnalysis>(*VF);
}

BasicBlock* VectorizedFunctionInfo::getDominator(BasicBlock* a, BasicBlock* b) {
    if (a == b) {
        FATAL(
            "Two predecessors that are the same...switch "
            "block with two conditions going to the same place?\n");
    } else if (doms->dominates(a, b) && doms->dominates(b, a)) {
        FATAL(" and " << b->getName() << " dominate each other?\n");
    } else if (doms->dominates(a, b)) {
        return a;
    } else if (doms->dominates(b, a)) {
        return b;
    } else {
        FATAL("Neither predecessor block dominates the other! "
              << "Structurization failed? " << a->getName() << " "
              << b->getName() << "\n");
    }
}

BasicBlock* VectorizedFunctionInfo::getPHIBackedge(PHINode* inst) {
    // A structurized CFG shouldn't have more than two
    assert(inst->getNumIncomingValues() == 2);

    // Get the predecessors
    BasicBlock* BB = inst->getParent();
    BasicBlock* a = inst->getIncomingBlock(0);
    BasicBlock* b = inst->getIncomingBlock(1);

    // At least one should be a forward edge
    bool BB_is_loop_header = loop_info->isLoopHeader(BB);
    BasicBlock* BB_loop_exiting_node = nullptr;
    if (BB_is_loop_header) {
        BB_loop_exiting_node = loop_info->getLoopFor(BB)->getExitingBlock();
    }
    bool a_fwd = a != BB_loop_exiting_node;
    bool b_fwd = b != BB_loop_exiting_node;
    assert(a_fwd || b_fwd);

    PRINT_HIGH("The current block is " << inst->getParent()->getName());
    if (BB_loop_exiting_node) {
        PRINT_HIGH("The exiting block for this loop is "
                   << BB_loop_exiting_node->getName());
    } else if (BB_is_loop_header) {
        PRINT_HIGH("This loop does not have an exiting node?");
        SmallVector<BasicBlock*> exiting_BBs;
        loop_info->getLoopFor(BB)->getExitingBlocks(exiting_BBs);
        for (BasicBlock* exiting : exiting_BBs) {
            PRINT_HIGH("  Exiting block " << exiting->getName());
        }
        assert(false);
    } else {
        PRINT_HIGH("The current block is not a loop header");
    }

    if (a_fwd) {
        PRINT_HIGH("The edge from " << a->getName() << " is a forward edge");
    } else {
        PRINT_HIGH("The edge from " << a->getName() << " is a backedge");
    }
    if (b_fwd) {
        PRINT_HIGH("The edge from " << b->getName() << " is a forward edge");
    } else {
        PRINT_HIGH("The edge from " << b->getName() << " is a backedge");
    }

    // If one of the two is a backedge, keep the PHI as is
    if (!a_fwd) {
        PRINT_HIGH("PHI has at least one backedge; keeping as PHI");
        return a;
    }
    if (!b_fwd) {
        PRINT_HIGH("PHI has at least one backedge; keeping as PHI");
        return b;
    }

    return nullptr;
}

Value* VectorizedFunctionInfo::getPHISelectMask(PHINode* phi,
                                                bool* is_inverted) {
    assert(phi->getNumIncomingValues() > 0);

    if (phi->getNumIncomingValues() == 1) {
        return nullptr;
    }

    assert(phi->getNumIncomingValues() == 2);
    BasicBlock* a = phi->getIncomingBlock(0);
    BasicBlock* b = phi->getIncomingBlock(1);

    BasicBlock* BB_backedge = getPHIBackedge(phi);
    if (BB_backedge) {
        // During mask calculation, we make sure that the backedge is always
        // the taken condition for the loop repeat branch
        BranchInst* term = cast<BranchInst>(BB_backedge->getTerminator());
        assert(term->isConditional());
        return term->getCondition();
    } else {
        // First, find the mask of the inner BB
        BasicBlock* dominator = getDominator(a, b);
        BasicBlock* then_block = (dominator == a) ? b : a;
        bool inv = (dominator == b) ? true : false;
        Value* mask = bb_masks[then_block].active_mask;
        PRINT_HIGH("Then block for " << *phi << " is " << then_block->getName()
                                     << " invert " << inv);
        if (is_inverted) {
            *is_inverted = inv;
        }
        return mask;
    }
}

void VectorizedFunctionInfo::verifyTransformedFunction() {
    // Verify that the function is still valid in the end
    PRINT_LOW("Verifying function...");
    PRINT_MID("");
    bool error = verifyFunction(*VF, &errs());
    if (error) {
        PRINT_LOW("\nFunction is:\n" << *VF);
        FATAL("\nVerification failed\n");
    }
}

VectorizedModuleInfo::VectorizedModuleInfo(Module* mod)
    : ctx(mod->getContext()), mod(mod) {}

}  // namespace ps
