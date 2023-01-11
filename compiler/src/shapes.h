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

#include <llvm/IR/Function.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "shape_calc.h"
#include "vectorize.h"

namespace ps {

extern unsigned shapes_verbosity_level;

class ShapesStep {
  public:
    ShapesStep(VectorizedFunctionInfo& vf_info);
    void calculate();

  private:
    VectorizedFunctionInfo& vf_info;
    ValueCache& value_cache;
    uint32_t num_lanes;

    template <typename T, typename... S>
    Shape tryTransform(std::vector<T> transforms, Shape sa, S... shapes);
    Shape transformKnownBases(std::function<z3::expr(z3::expr)> f, Shape sa);
    Shape transformKnownBases(std::function<z3::expr(z3::expr, z3::expr)> f,
                              Shape sa, Shape sb);
    Shape transformKnownBases(
        std::function<z3::expr(z3::expr, z3::expr, z3::expr)> f, Shape sa,
        Shape sb, Shape sc);

    void calculateShape(std::unordered_set<llvm::Instruction*>& work_queue,
                        llvm::Instruction* I, bool allow_overwrite = false);
    Shape calculateShapeBinaryOp(llvm::BinaryOperator* inst);
    Shape calculateShapeCall(llvm::CallInst* inst);
    Shape calculateShapeGEP(llvm::GetElementPtrInst* inst);
    Shape calculateShapeCmp(llvm::ICmpInst* inst);
    Shape calculateShapeLoad(llvm::LoadInst* inst);
    Shape calculateShapePHI(llvm::PHINode* inst);
    Shape calculateShapeSelect(llvm::SelectInst* select);
    Shape calculateShapeTrunc(llvm::TruncInst* trunc);
    Shape calculateShapeUIToFP(llvm::UIToFPInst* uitofp);
    Shape calculateShapeExt(llvm::Instruction* ext, bool is_signed);

    void arrayLayoutOpt();
    bool analyzeUses(llvm::Instruction* inst);
    llvm::Instruction* generateOptInsts(
        llvm::AllocaInst* inst,
        std::set<std::pair<llvm::Instruction*, llvm::Instruction*>>& toReplace);
    void insertOptInsts(
        std::set<std::pair<llvm::Instruction*, llvm::Instruction*>>& toReplace);

    void calulateFinalMemInstMappedShapes();
    void printShapes();

    unsigned getValueSizeBits(llvm::Value* v);
    unsigned getValueSizeBytes(llvm::Value* v);
    unsigned getBaseValueSizeBytes(llvm::Value* v);

    struct GlobalValuePlusOffset {
        llvm::GlobalValue* gv;
        int64_t offset;
    };
    llvm::GlobalValue* getGlobalValueFromExpr(z3::expr base);
    GlobalValuePlusOffset getGlobalValuePlusOffsetFromExpr(z3::expr base);
    std::unordered_map<std::string, llvm::GlobalValue*> shape_constants;
};

}  // namespace ps
