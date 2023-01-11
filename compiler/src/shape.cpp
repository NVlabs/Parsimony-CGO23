/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "shape.h"

namespace ps {

std::string Shape::toString(bool symbolic_indices) const {
    std::stringstream s;
    s << "Shape: " << std::setw(8);

    if (isUniform()) {
        s << "Uniform ";
    } else if (isUnknown()) {
        s << "Unknown ";
    } else if (isVarying()) {
        s << "Varying ";
    } else if (isIndexed()) {
        s << "Indexed ";
    } else if (isNone()) {
        s << "None ";
    }

    if (isIndexed()) {
        s << ", {";
        for (unsigned i = 0; i < indices.size(); i++) {
            if (i > 0) {
                s << ",";
            }
            if (symbolic_indices) {
                s << indices[i].simplify().to_string();
            } else {
                s << std::to_string((int64_t)getIndexAsInt(i));
            }
        }

        if (hasConstantBase()) {
            s << "}, base " << (int64_t)getConstantBase();
        } else {
            s << "}, base " << base.simplify().to_string();
        }
        s << ", width " << base.get_sort().bv_size();
    }

    return s.str();
}

int64_t Shape::getMaxIndex() {
    assert(type == INDEXED);
    int64_t max = INT64_MIN;
    for (unsigned i = 0; i < indices.size(); i++) {
        int64_t idx = getIndexAsInt(i);
        max = std::max(idx, max);
    }
    return max;
}

int64_t Shape::getMinIndex() {
    assert(type == INDEXED);
    int64_t min = INT64_MAX;
    for (unsigned i = 0; i < indices.size(); i++) {
        int64_t idx = getIndexAsInt(i);
        min = std::min(idx, min);
    }
    return min;
}

bool Shape::isGangPacked(size_t elem_size) {
    if (type != INDEXED) {
        return false;
    }
    int64_t min = getMinIndex();
    int64_t max = getMaxIndex();
    min = ceilDiv(min, (int64_t)elem_size);
    max = ceilDiv(max, (int64_t)elem_size);
#if 1
    const int max_factor = 4;
    if (max - min < (int64_t)indices.size() * max_factor) {
        return true;
    }
#else
    if (min >= 0 && max < (int64_t)indices.size()) {
        return true;
    }
#endif
    return false;
}

z3::context Shape::ctx_for_invalid_bases;

}  // namespace ps
