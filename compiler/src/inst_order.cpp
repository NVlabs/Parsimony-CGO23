/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "inst_order.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>

using namespace llvm;

namespace ps {

unsigned inst_order_verbosity_level;
[[maybe_unused]] static unsigned& verbosity_level = inst_order_verbosity_level;

bool InstructionOrderStep::OperandDominatedByUser(Instruction* operand,
                                                  Instruction* user) {
    if (vf_info.doms->dominates(user, operand)) {
        if (!isa<PHINode>(user)) {
            FATAL("Unexpected back-edge\nfrom "
                  << *operand << "\nto non PHINode " << *user);
        }

        return true;
    }

    return false;
}

void InstructionOrderStep::calculate() {
    // PRINT_MID("\n");
    // PRINT_LOW("Calculating instruction order:");
    // PRINT_MID("");

    // From each instruction to any predecessors that have not yet been placed
    // into the instruction order
    std::unordered_map<Instruction*, std::unordered_set<Instruction*>>
        pending_defs;

    // From each instruction to any other instructions that use it
    std::unordered_map<Instruction*, std::unordered_set<Instruction*>> uses;

    // Instructions for which all predecessors have already been placed into the
    // instruction order
    std::unordered_set<Instruction*> insts_with_no_pending_deps;

    for (BasicBlock& BB : *vf_info.VF) {
        for (Instruction& I : BB) {
            // if (verbosity_level) {
            // PRINT_HIGH("Analyzing " << I);
            //}

            for (Use& U : I.operands()) {
                Instruction* operand = dyn_cast<Instruction>(U.get());
                if (!operand) {
                    continue;
                }

                if (OperandDominatedByUser(operand, &I)) {
                    // if (verbosity_level) {
                    //    PRINT_HIGH("  Ignoring backedge from " << *operand
                    //                                           << " to " <<
                    //                                           I);
                    //}
                } else {
                    // if (verbosity_level) {
                    //    PRINT_HIGH(" depends on " << *operand);
                    //}
                    pending_defs[&I].insert(operand);
                    uses[operand].insert(&I);
                }
            }

            // FIXME "mask"
            if (I.getName().find("mask") == std::string::npos) {
                Instruction* active_mask = dyn_cast<Instruction>(
                    vf_info.bb_masks[I.getParent()].active_mask);
                if (active_mask) {
                    pending_defs[&I].insert(active_mask);
                    uses[active_mask].insert(&I);
                    // if (verbosity_level) {
                    //    PRINT_HIGH(" depends on BB " << BB.getName()
                    //                                 << " active mask "
                    //                                << *active_mask);
                    //}
                }
            }

            // Special case: PHIs will eventually be made to depend on some
            // predecessor BBs' active masks
            PHINode* phi = dyn_cast<PHINode>(&I);
            if (phi && phi->getNumIncomingValues() > 1) {
                Value* mask = vf_info.getPHISelectMask(phi);
                assert(mask);
                if (mask && !vf_info.getPHIBackedge(phi)) {
                    // Only treat the active_mask as a dependency if it is
                    // calculated by an instruction, not if it's a constant
                    Instruction* mask_inst = dyn_cast<Instruction>(mask);
                    if (mask_inst) {
                        pending_defs[&I].insert(mask_inst);
                        uses[mask_inst].insert(&I);
                    }
                }
            }

            // If none of the above were actual dependencies, this instruction
            // has no dependencies, and can be inserted into the order right
            // from the start
            if (pending_defs.find(&I) == pending_defs.end()) {
                // if (verbosity_level) {
                //    PRINT_HIGH(" has no dependencies");
                //}
                insts_with_no_pending_deps.insert(&I);
            } else {
                // if (verbosity_level) {
                //    PRINT_HIGH(" has " << pending_defs[&I].size()
                //                       << " dependencies");
                //}
                assert(!pending_defs[&I].empty());
            }
        }
    }

    while (!insts_with_no_pending_deps.empty()) {
        Instruction* I = *insts_with_no_pending_deps.begin();
        // if (verbosity_level) {
        //    PRINT_MID("Choosing " << *I);
        //}
        insts_with_no_pending_deps.erase(I);
        instruction_order.push_back(I);

        for (Instruction* use : uses[I]) {
            pending_defs[use].erase(I);
            // if (verbosity_level) {
            //    PRINT_HIGH(" has " << pending_defs[use].size()
            //                      << " pending deps");
            // }
            if (pending_defs[use].empty()) {
                insts_with_no_pending_deps.insert(use);
                pending_defs.erase(use);
            }
        }
    }

    if (!pending_defs.empty()) {
        PRINT_ALWAYS(
            "Cycle found during instruction dependency order "
            "calculation");
        for (auto i : pending_defs) {
            PRINT_HIGH(" depends on:");
            for (auto j : i.second) {
                PRINT_HIGH("  " << *j);
            }
        }
        assert(false);
    }
}

}  // namespace ps
