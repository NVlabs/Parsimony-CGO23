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
#include <cassert>
#include <unordered_map>
#include <vector>

#include "shape.h"

namespace ps {

struct UnaryShapeTransform {
    std::string name;
    std::function<z3::expr(z3::expr)> f_expr;
    std::function<z3::expr(unsigned, Shape)> f_proposed_index;
    std::vector<std::function<z3::expr(Shape)>> assumptions;
};

struct BinaryShapeTransform {
    std::string name;
    std::function<z3::expr(z3::expr, z3::expr)> f_expr;
    std::function<z3::expr(unsigned, Shape, Shape)> f_proposed_index;
    std::vector<std::function<z3::expr(Shape, Shape)>> assumptions;
};

struct KnownTransforms {
    KnownTransforms();
    std::unordered_map<std::string, BinaryShapeTransform> binary;
    UnaryShapeTransform sext(unsigned target_width);
    UnaryShapeTransform trunc(unsigned target_width);
    UnaryShapeTransform zext(unsigned target_width);
};

extern KnownTransforms known_transforms;

}  // namespace ps
