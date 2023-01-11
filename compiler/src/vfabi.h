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

#include <string>
#include <vector>

#include <llvm/IR/Module.h>

namespace ps {

struct VFABIShape {
    static VFABIShape Varying() { return {true, 0, 0}; }
    static VFABIShape Strided(int stride, unsigned alignment = 0) {
        return {false, stride, alignment};
    }
    static VFABIShape Uniform(unsigned alignment = 0) {
        return {false, 0, alignment};
    }

    bool is_varying;
    int stride;
    unsigned alignment;
};

struct VFABI {
    VFABI()
        : is_entry_point(false),
          is_declare_spmd(false),
          isa(""),
          mask(false),
          vlen(0),
          return_shape(VFABIShape::Varying()),
          scalar_name(""),
          mangled_name("") {}

    bool is_entry_point;
    bool is_declare_spmd;

    std::string isa;
    bool mask;
    unsigned vlen;
    // TODO if we ever need it: ref, val, uval
    // TODO if we ever need it: "runtime linear"
    std::vector<VFABIShape> parameters;
    VFABIShape return_shape;
    std::string scalar_name;
    std::string mangled_name;

    std::string toString() const;
};

void getFunctionVFABIs(llvm::Function* f, std::vector<VFABI>& vfabis);
bool getFunctionAttributeVFABI(std::string attribute_string, VFABI& vfabi);

extern unsigned vfabi_verbosity_level;

}  // namespace ps
