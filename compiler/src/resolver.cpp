/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "resolver.h"

#include <llvm/Analysis/VectorUtils.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>

#include "broadcast.h"
#include "utils.h"

using namespace llvm;

namespace ps {

unsigned resolver_verbosity_level;
[[maybe_unused]] static unsigned& verbosity_level = resolver_verbosity_level;

FunctionResolution FunctionResolver::getBestVFABIMatch(
    std::vector<FunctionResolution>& resolutions, VFABI& desired) {
    PRINT_HIGH("Considering " << resolutions.size() << " resolutions");

    // Find all functionally compatible VFABIs
    std::vector<FunctionResolution> candidates;
    for (FunctionResolution& resolution : resolutions) {
        PRINT_HIGH("Considering resolution " << resolution.vfabi.toString());

        VFABI& vfabi = resolution.vfabi;
        if (vfabi.isa != desired.isa ||
            vfabi.mask != desired.mask ||  // could relax if needed
            vfabi.vlen != desired.vlen) {  // could relax this as well
            PRINT_HIGH("VFABI " << vfabi.toString() << " is incompatible");
            continue;
        }

        bool incompatible = false;
        if (vfabi.parameters.size() != desired.parameters.size()) {
            FATAL("Provided argument count does not match "
                  << "expected argument count");
        }
        for (unsigned i = 0; i < vfabi.parameters.size(); i++) {
            if (desired.parameters[i].is_varying &&
                !vfabi.parameters[i].is_varying) {
                PRINT_HIGH("VFABI " << vfabi.toString()
                                    << " is incompatible due to parameter "
                                    << i);
                incompatible = true;
                break;
            }
            unsigned desired_alignment =
                1;  // FIXME desired.parameters[i].alignment;
            unsigned provided_alignment =
                1;  // FIXME vfabi.parameters[i].alignment;

            if (provided_alignment > 0 &&
                !isMultipleOf(desired_alignment, provided_alignment)) {
                PRINT_HIGH("VFABI " << vfabi.toString()
                                    << " is incompatible due to parameter " << i
                                    << " alignment");
                incompatible = true;
                break;
            }
        }
        if (incompatible) {
            continue;
        }

        PRINT_HIGH("VFABI " << vfabi.toString() << " is compatible");
        candidates.push_back(resolution);
    }

    if (candidates.empty()) {
        return {nullptr, VFABI()};
    }

    // Pick the best candidate
    // For now, there should only be one candidate, because so far we are
    // checking for exact matches of isa, mask, and vlen
    if (candidates.size() > 1) {
        FATAL("More than one legal function resolution candidate");
    }
    return candidates[0];
}

void FunctionResolver::add(Function* f, FunctionResolution resolution) {
    PRINT_HIGH("Adding function " << f << " " << f->getName() << " resolution: "
                                  << resolution.function->getName() << " ABI "
                                  << resolution.vfabi.toString());

    resolver_map[f].push_back(resolution);
}

FunctionResolver::PsimApiEnum FunctionResolver::getPsimApiEnum(
    Function* f) {
    for (auto& a : PsimApiEnumStrMap) {
        if (isBaseFunctionName(f, a.second)) {
            return a.first;
        }
    }
    return PSIM_API_NONE;
}

FunctionResolution FunctionResolver::get(Function* f, VFABI& desired) {
    PRINT_HIGH("Resolving function " << f << " " << f->getName()
                                     << " for VFABI " << desired.toString());
    auto it = resolver_map.find(f);

    if (it == resolver_map.end()) {
        PRINT_HIGH("Resolver cache miss");
        return {nullptr, VFABI()};
    }

    PRINT_HIGH("Resolver cache hit");
    return getBestVFABIMatch(it->second, desired);
}

}  // namespace ps
