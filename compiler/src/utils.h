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

#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/raw_ostream.h>
#include <z3++.h>

#define DEBUG_LEVEL(x, n)       \
    if (verbosity_level >= n) { \
        x;                      \
    }

#define PRINT_LEVEL(x, n) \
    DEBUG_LEVEL(          \
        llvm::errs() << __FILE__ << ":" << __LINE__ << ": " << x << "\n", n)

#define DEBUG_HIGH(x) DEBUG_LEVEL(x, 3)
#define DEBUG_MID(x) DEBUG_LEVEL(x, 2)
#define DEBUG_LOW(x) DEBUG_LEVEL(x, 1)

#define PRINT_HIGH(x) PRINT_LEVEL(x, 3)
#define PRINT_MID(x) PRINT_LEVEL(x, 2)
#define PRINT_LOW(x) PRINT_LEVEL(x, 1)
#define PRINT_ALWAYS(x) \
    llvm::errs() << __FILE__ << ":" << __LINE__ << ": " << x << "\n"

#define WARNING(msg) llvm::errs() << "WARNING: " << msg << "\n";

#define FATAL(msg)                              \
    do {                                        \
        fflush(stdout);                         \
        PRINT_ALWAYS("FATAL: " << msg << "\n"); \
        abort();                                \
    } while (0)

#define ASSERT(cond, msg)                                               \
    {                                                                   \
        if (!(cond)) {                                                  \
            fflush(stdout);                                             \
            PRINT_ALWAYS("ASSERT fail: " #cond << ": " << msg << "\n"); \
            abort();                                                    \
        }                                                               \
    }

namespace ps {

typedef struct global_opts_t {
    bool add_prints;
    bool error_on_warn;
    bool ignore_warn_set;
    int scalable_size;
} global_opts_t;

extern global_opts_t global_opts;

inline std::string valueString(llvm::Value* V) {
    if (!V) {
        return "nullptr";
    }
    std::string s;
    llvm::raw_string_ostream OS(s);
    V->print(OS);
    return s;
}

static inline bool isBaseFunctionName(llvm::Function* f, std::string name) {
    if (f && f->hasName()) {
        std::string demangled = llvm::demangle(f->getName().str().c_str());
        /*TODO: trim demangled function name to pick only base name */
        if (demangled.find(name) != std::string::npos) {
            return true;
        }
    }
    return false;
}

inline bool isMultipleOf(unsigned a, unsigned b) { return a / b * b == a; }

inline unsigned roundUp(unsigned a, unsigned b) {
    return ((a + (b - 1)) / b) * b;
}

template <typename T>
inline T ceilDiv(T a, T b) {
    return ((a + (b - 1)) / b);
}

inline bool isPowerOfTwo(uint64_t a) { return (a & (a - 1)) == 0; }
inline z3::expr exprIsPowerOfTwo(z3::expr e) { return (e & (e - 1)) == 0; }

llvm::Type* vectorizeType(llvm::Type* ty, unsigned num_lanes);
llvm::ElementCount getElementCount(unsigned num_lanes);
std::vector<uint64_t> getValuesFromGlobalConstant(llvm::Value* value);
std::string getDebugLocStr(llvm::Instruction* inst, int leading_zeros = 0);
}  // namespace ps
