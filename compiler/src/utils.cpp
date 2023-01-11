/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "utils.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/DebugLoc.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <vector>

#include "utils.h"
#include "vfabi.h"

using namespace llvm;

namespace ps {

global_opts_t global_opts;

ElementCount getElementCount(unsigned num_lanes) {
    if (global_opts.scalable_size) {
        return LinearPolySize<ElementCount>::getScalable(
            num_lanes / global_opts.scalable_size);
    } else {
        return LinearPolySize<ElementCount>::getFixed(num_lanes);
    }
}

Type* vectorizeType(Type* ty, unsigned num_lanes) {
    if (ty->isVectorTy()) {
        return ty;
    }

    if (ty == Type::getVoidTy(ty->getContext())) {
        return ty;
    }

    if (FunctionType* FT = dyn_cast<FunctionType>(ty)) {
        assert(!FT->isVarArg());

        Type* return_type = vectorizeType(FT->getReturnType(), num_lanes);

        std::vector<Type*> param_types;
        for (auto& i : FT->params()) {
            param_types.push_back(vectorizeType(i, num_lanes));
        }

        return FunctionType::get(return_type, param_types, false);
    }

    if (ty->isSingleValueType()) {
        return VectorType::get(ty, getElementCount(num_lanes));
    }

    FATAL("Don't know how to vectorize type '" << *ty << "'!\n");
}

static void getValuesFromGlobalConstantInner(Constant* c,
                                             std::vector<uint64_t>& values,
                                             std::string indent = "  ") {
    // PRINT_ALWAYS(indent << "Constant " << *c);

    int elem = 0;
    while (1) {
        Constant* e = c->getAggregateElement(elem);
        if (e) {
            // PRINT_ALWAYS(indent << "ConstantElem[" << elem << "] " << *e);
            getValuesFromGlobalConstantInner(e, values, indent + "  ");
            elem++;
        } else {
            break;
        }

        ConstantAggregateZero* caz = dyn_cast<ConstantAggregateZero>(e);
        ConstantInt* cint = dyn_cast<ConstantInt>(e);
        if (cint) {
            // PRINT_ALWAYS(indent << "ConstantInt " << *cint);
            int64_t cval = cint->getValue().getZExtValue();
            values.push_back(cval);
        } else if (caz) {
            // PRINT_ALWAYS(indent << "ConstantAggregateZero " << *caz);
            for (uint32_t i = 0; i < caz->getElementCount().getFixedValue();
                 i++) {
                values.push_back(0);
            }
        }
    }
}

std::vector<uint64_t> getValuesFromGlobalConstant(Value* value) {
    std::vector<uint64_t> values;

    GlobalVariable* data = dyn_cast<GlobalVariable>(value);
    if (data) {
        Constant* c = data->getInitializer();
        getValuesFromGlobalConstantInner(c, values);
    }
    return values;
}

std::string getDebugLocStr(Instruction* inst, int leading_zeros) {
    const llvm::DebugLoc& debugInfo = inst->getDebugLoc();
    if (!debugInfo) {
        return "<no line info available>";
    }
    std::string str;
    llvm::raw_string_ostream rso(str);
    debugInfo.print(rso);
    return str;
}

}  // namespace ps
