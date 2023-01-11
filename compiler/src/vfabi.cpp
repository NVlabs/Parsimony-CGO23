/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "vfabi.h"
#include "utils.h"

using namespace llvm;

namespace ps {

unsigned vfabi_verbosity_level;
[[maybe_unused]] static unsigned& verbosity_level = vfabi_verbosity_level;

bool decodeVFABI(std::string attribute_string, VFABI& vfabi) {
    PRINT_HIGH("Parsing VFABI string " << attribute_string);

    // _ZGV
    if (attribute_string.substr(0, 4) != "_ZGV") {
        return false;
    }
    assert(attribute_string.size() > 9);

    // ISA
    vfabi.isa = attribute_string.substr(4, 1);
    assert(vfabi.isa != "_");  // as in "__LLVM__"

    // Mask
    char mask = attribute_string[5];
    assert(mask == 'M' || mask == 'N');
    vfabi.mask = (mask == 'M');

    // vlen
    vfabi.vlen = std::stoi(attribute_string.substr(6));

    // parameters
    size_t i;
    for (i = 7; i < attribute_string.size(); i++) {
        // skip the rest of the 'vfabi.vlen' digits
        if (!isdigit(attribute_string[i])) {
            break;
        }
    }
    for (; i < attribute_string.size(); i++) {
        char c = attribute_string[i];
        if (c == '_') {
            break;
        }

        PRINT_HIGH("Parsing character '" << c << "'");
        switch (c) {
            case 'a': {
                // Update the last parameter to use the given alignment
                assert(isdigit(attribute_string[i + 1]));
                assert(!vfabi.parameters.empty());
                size_t digits;
                vfabi.parameters.back().alignment =
                    std::stoi(&attribute_string[i + 1], &digits);
                i += digits;
            } break;
            case 'l': {
                if (isdigit(attribute_string[i + 1])) {
                    size_t digits;
                    vfabi.parameters.push_back(VFABIShape::Strided(
                        std::stoi(&attribute_string[i + 1], &digits)));
                    i += digits;
                } else if (attribute_string[i + 1] == 's') {
                    FATAL("Stride as argument not supported");
                } else {
                    vfabi.parameters.push_back(VFABIShape::Strided(1));
                }
            } break;
            case 'u':
                vfabi.parameters.push_back(VFABIShape::Uniform());
                break;
            case 'v':
                vfabi.parameters.push_back(VFABIShape::Varying());
                break;
            default:
                assert(false);
        }
    }
    assert(i < attribute_string.size());

    // scalar name
    vfabi.scalar_name = attribute_string.substr(i);
    vfabi.mangled_name = attribute_string;
    // Done!
    return true;
}

void getFunctionVFABIs(llvm::Function* f, std::vector<VFABI>& vfabis) {
    for (auto& aset : f->getAttributes()) {
        for (auto& a : aset) {
            std::string attribute_string = a.getAsString();

            VFABI vfabi;
            bool success = getFunctionAttributeVFABI(attribute_string, vfabi);
            if (success) {
                vfabis.push_back(vfabi);

                if (vfabi.is_entry_point) {
                    // Don't keep parsing more attributes in this case; we
                    // already found the magic indicator
                    return;
                }
            }
        }
    }
}

bool getFunctionAttributeVFABI(std::string attribute_string, VFABI& vfabi) {
    // remove outer quotes
    if (attribute_string[0] == '"') {
        attribute_string =
            attribute_string.substr(1, attribute_string.size() - 2);
    }

    // remove double underscore
    auto double_u = attribute_string.find("__");
    if (double_u != std::string::npos) {
        attribute_string = attribute_string.replace(double_u, 2, "_");
    }

    // Actually decode
    return decodeVFABI(attribute_string, vfabi);
}

std::string VFABI::toString() const {
    std::string s;

    if (is_declare_spmd) {
        s += "_spmd";
    }

    s += "_ZGV";
    s += isa;
    s += mask ? 'M' : 'N';
    s += std::to_string(vlen);

    for (const VFABIShape& p : parameters) {
        if (!p.is_varying && p.stride == 0) {
            s += 'u';
        } else if (!p.is_varying) {
            s += 'l' + std::to_string(p.stride);
        } else {
            s += 'v';
        }

        if (p.alignment != 0) {
            s += 'a' + std::to_string(p.alignment);
        }
    }

    s += "_" + scalar_name;
    return s;
}

}  // namespace ps
