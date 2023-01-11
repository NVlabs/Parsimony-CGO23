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

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>

#include <unordered_map>

#include "shape.h"

namespace ps {

extern unsigned value_cache_verbosity_level;

struct VectorizedFunctionInfo;

class ValueCache {
  public:
    ValueCache(llvm::Function* VF, unsigned num_lanes,
               VectorizedFunctionInfo* vf_info)
        : VF(VF),
          ctx(VF->getContext()),
          num_lanes(num_lanes),
          vf_info(vf_info) {}

    bool has(llvm::Value* value) const;

    void setVectorValue(llvm::Value* value, llvm::Value* vector_value);
    void setScalarValue(llvm::Value* value, llvm::Value* scalar_value);
    void setShape(llvm::Value* value, Shape shape, bool overwrite = false);
    void setToBeDeleted(llvm::Value* value);
    void setArrayLayoutOpt(llvm::Value* value);

    llvm::Value* getScalarValue(llvm::Value* value);
    llvm::Value* getVectorValue(llvm::Value* value);
    Shape getShape(llvm::Value* value);
    bool getArrayLayoutOpt(llvm::Value* value);
    MemInstMappedShape getMemInstMappedShape(llvm::Instruction* inst);
    void setMemInstMappedShape(llvm::Instruction* inst,
                               MemInstMappedShape minst_mapping);
    void deleteObsoletedInsts();

    std::string getConstName(llvm::Value* value);
    llvm::Value* genConstVect(llvm::Constant* C, llvm::IRBuilder<>& builder);

  private:
    struct ValueCacheEntry {
        ValueCacheEntry(llvm::Value* scalar_value, Shape shape)
            : scalar_value(scalar_value),
              vector_value(nullptr),
              shape(shape),
              to_be_deleted(false),
              already_deleted(false),
              arrayLayoutOpt(false) {}

        // the non-vectorized IR representation of uniform and strided shapes.
        // for strided shapes, this is the base element.
        llvm::Value* scalar_value;

        // the IR representation of varying shapes, or vectorized versions of
        // uniform or strided shapes
        llvm::Value* vector_value;

        Shape shape;

        // mapped shape of memory instruction (if memory instruction)
        MemInstMappedShape minst_mapping;

        bool to_be_deleted;
        bool already_deleted;
        bool arrayLayoutOpt;
    };

    ValueCacheEntry& get(llvm::Value* value);

    llvm::Function* VF;
    llvm::LLVMContext& ctx;
    unsigned num_lanes;
    VectorizedFunctionInfo* vf_info;
    std::unordered_map<llvm::Value*, ValueCacheEntry> entries;
    void deleteInst(llvm::Instruction* I, unsigned prefix = 0);

    std::string unknown_const_name_string = "$psv";
    unsigned unknown_const_name_counter = 0;
};

}  // namespace ps
