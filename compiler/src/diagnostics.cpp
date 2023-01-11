/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <llvm/Demangle/Demangle.h>

#include "diagnostics.h"

using namespace llvm;

namespace ps {

unsigned diagnostics_verbosity_level;
[[maybe_unused]] static unsigned& verbosity_level = diagnostics_verbosity_level;

////////////////////////////////////////////////////////////////////////////////

void printVector(std::vector<std::string>& strings, std::string before,
                 std::string after, std::string indent = "") {
    if (!strings.empty()) {
        llvm::errs() << "    " << before << " " << strings.size() << " "
                     << after << "\n";

        DEBUG_LOW({
            for (const std::string& s : strings) {
                llvm::errs() << "        " << indent << s << "\n";
            }
        })
    }
}

void printSet(std::set<std::string>& strings, std::string before,
              std::string after, std::string indent = "",
              bool always_expand = false) {
    if (!strings.empty()) {
        llvm::errs() << "    " << before << " " << strings.size() << " "
                     << after << "\n";

        if (always_expand || verbosity_level > 0) {
            for (const std::string& s : strings) {
                llvm::errs() << "        " << indent << s << "\n";
            }
        }
    }
}

void printDiagnostics(VectorizedFunctionInfo* vf_info) {
    bool hasDiagnostics =
        !vf_info->diagnostics.unhandled_shape_opcodes.empty() ||
        !vf_info->diagnostics.gathers.empty() ||
        !vf_info->diagnostics.scatters.empty() ||
        !vf_info->diagnostics.scalarized_called_functions.empty() ||
        !vf_info->diagnostics.function_pointer_calls.empty() ||
        !vf_info->diagnostics.unoptimized_allocas.empty();
    if (!hasDiagnostics || verbosity_level == 0) {
        return;
    }

    llvm::errs() << "----------------------------------------------------------"
                    "----------------------\n";
    llvm::errs() << "Diagnostics for function "
                 << llvm::demangle(vf_info->vfabi.scalar_name)
                 << ": gang size = " << vf_info->vfabi.vlen
                 << "; ABI = " << vf_info->vfabi.toString() << "\n";
    /* comment for now, maybe in future we want to see the shape of the
     * parameters */
    /*
    int cnt = 0;
    for (auto p : vf_info->vfabi.parameters) {
        llvm::errs() << "    Arg[" << std::to_string(cnt++)
                     << "] = " << p.toString() << "\n";
    }
    */

    if (!vf_info->diagnostics.unhandled_shape_opcodes.empty()) {
        llvm::errs() << "    Shapes not handled during shape analysis: ";

        bool first = true;
        for (const std::string& o :
             vf_info->diagnostics.unhandled_shape_opcodes) {
            if (first) {
                first = false;
            } else {
                llvm::errs() << ", ";
            }
            llvm::errs() << o;
        }
        llvm::errs() << "\n";

        DEBUG_LOW({
            for (std::string& s : vf_info->diagnostics.unhandled_shape_insts) {
                llvm::errs() << "        " << s << "\n";
            }
        })
    }

    for (auto s : vf_info->diagnostics.gathers) {
        printVector(s.second, "Emitted",
                    "gather instructions of size " + std::to_string(s.first) +
                        " bytes");
    }
    for (auto s : vf_info->diagnostics.scatters) {
        printVector(s.second, "Emitted",
                    "scatter instructions of size " + std::to_string(s.first) +
                        " bytes");
    }
    printSet(vf_info->diagnostics.scalarized_called_functions,
             "Emitted scalarized calls to", "functions", "  ", true);
    printVector(vf_info->diagnostics.function_pointer_calls,
                "Emitted scalarized calls to", "function pointers");
    printVector(vf_info->diagnostics.unoptimized_allocas, "Emitted",
                "unoptimized allocas");

    llvm::errs() << "----------------------------------------------------------"
                    "----------------------\n";
}

}  // namespace ps
