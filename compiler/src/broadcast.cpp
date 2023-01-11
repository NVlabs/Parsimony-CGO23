/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "broadcast.h"
#include "utils.h"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>

using namespace llvm;

namespace ps {

unsigned broadcast_verbosity_level;
[[maybe_unused]] static unsigned& verbosity_level = broadcast_verbosity_level;

Value* createStrideConstant(Constant* v, unsigned num_lanes, int64_t stride) {
    std::vector<Constant*> lanes;

    if (stride == 0) {
        for (int64_t i = 0; i < num_lanes; i++) {
            lanes.push_back(v);
        }
    } else {
        Type* ty = v->getType();
        ConstantInt* CI = dyn_cast<ConstantInt>(v);
        if (!CI) {
            FATAL("Unexpected constant " << *v << " of type " << *ty);
        }

        int64_t base = CI->getZExtValue();
        for (int64_t i = 0; i < num_lanes; i++) {
            lanes.push_back(ConstantInt::get(ty, base + i * stride));
        }
    }

    return ConstantVector::get(lanes);
}

}  // namespace ps
