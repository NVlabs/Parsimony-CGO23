/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "live_out.h"

#include <cassert>
#include <vector>

#include <llvm/IR/Module.h>

#include "vectorize.h"

using namespace llvm;

namespace ps {

unsigned live_out_verbosity_level;
[[maybe_unused]] static unsigned& verbosity_level = live_out_verbosity_level;

LiveOutPHIsStep::LiveOutPHIsStep(VectorizedFunctionInfo& vf_info)
    : vf_info(vf_info) {}

void LiveOutPHIsStep::calculate() {
    for (Loop* loop : vf_info.loop_info->getLoopsInPreorder()) {
        std::vector<Instruction*> live_out;
        BasicBlock* head = loop->getHeader();
        BasicBlock* tail = loop->getExitingBlock();

        assert(head->hasNPredecessors(2));
        auto it = pred_begin(head);
        BasicBlock* predecessor_a = *it;
        it++;
        BasicBlock* predecessor_b = *it;
        BasicBlock* preheader;
        if (predecessor_a == tail) {
            preheader = predecessor_b;
        } else {
            preheader = predecessor_a;
            assert(predecessor_b == tail);
        }

        for (BasicBlock* BB : loop->blocks()) {
            // Only look for defs for which 'loop' is the innermost loop
            Loop* BB_inner_loop = vf_info.loop_info->getLoopFor(BB);
            if (BB_inner_loop != loop) {
                continue;
            }

            for (Instruction& I : *BB) {
                for (User* user : I.users()) {
                    Instruction* user_inst = cast<Instruction>(user);
                    if (!loop->contains(user_inst->getParent())) {
                        live_out.push_back(&I);
                        PRINT_HIGH("Loop " << loop->getName()
                                           << " has live out " << I << "\n");
                        break;
                    }
                }
            }
        }

        for (Instruction* I : live_out) {
            PHINode* live_in = PHINode::Create(
                I->getType(), 2, I->getName() + "_livein", &head->front());
            SelectInst* live_out = SelectInst::Create(
                vf_info.bb_masks[tail].active_mask, I, live_in,
                I->getName() + "_liveout", tail->getTerminator());
            // Replace uses of I outside the loop with live_out
            I->replaceUsesWithIf(live_out, [I, loop](Use& U) {
                Instruction* inst = cast<Instruction>(U.getUser());
                bool replace = (inst != I) && !loop->contains(inst);
                // if (replace) {
                //     PRINT_HIGH("Replacing out-of-loop use " << *inst <<
                //     "\n");
                // } else {
                //     PRINT_HIGH("Not replacing in-loop use " << *inst <<
                //     "\n");
                // }
                return replace;
            });
            live_out->setOperand(1, I);
            live_in->addIncoming(PoisonValue::get(I->getType()), preheader);
            live_in->addIncoming(live_out, tail);
        }
    }

    PRINT_HIGH("After inserting loop live in/out:\n" << *vf_info.VF << "\n");
}

}  // namespace ps
