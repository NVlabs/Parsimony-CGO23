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

extern unsigned live_out_verbosity_level;

class LiveOutPHIsStep {
  public:
    LiveOutPHIsStep(VectorizedFunctionInfo& vf_info);
    void calculate();

  private:
    VectorizedFunctionInfo& vf_info;
};

}  // namespace ps
