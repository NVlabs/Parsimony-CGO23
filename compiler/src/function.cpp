/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "function.h"
#include "inst_order.h"
#include "live_out.h"
#include "mask.h"
#include "prints.h"
#include "shapes.h"
#include "transform.h"

using namespace llvm;

namespace ps {

unsigned function_verbosity_level;
[[maybe_unused]] static unsigned& verbosity_level = function_verbosity_level;

FunctionVectorizer::FunctionVectorizer(VectorizedFunctionInfo& vf_info)
    : vf_info(vf_info) {}

void FunctionVectorizer::vectorize() {
    vf_info.getAnalyses();

    MasksStep(vf_info).calculate();
    LiveOutPHIsStep(vf_info).calculate();
    InstructionOrderStep(vf_info).calculate();
    ShapesStep(vf_info).calculate();
    TransformStep(vf_info).transform();

    vf_info.verifyTransformedFunction();

    PRINT_LOW("Done vectorizing " << vf_info.VF->getName() << "\n");
    PRINT_MID(*vf_info.VF << "\n");

    if (global_opts.add_prints) {
        AddPrintsStep(vf_info).addPrints();
        vf_info.verifyTransformedFunction();
    }
}

}  // namespace ps
