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

#include <llvm/IR/Constants.h>
#include <llvm/IR/Value.h>

namespace ps {

llvm::Value* createStrideConstant(llvm::Constant* v, unsigned num_lanes,
                                  int64_t stride);
extern unsigned broadcast_verbosity_level;

}  // namespace ps
