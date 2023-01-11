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

#include <llvm/IR/Value.h>
#include <unordered_map>
#include "vectorize.h"
#include "vfabi.h"

namespace ps {

extern unsigned module_verbosity_level;

class ModuleVectorizer {
  public:
    ModuleVectorizer(VectorizedModuleInfo& vm_info) : vm_info(vm_info) {}
    void initialize();
    void vectorizeFunctions();
    void writeToFile(const std::string& fileName);

  private:
    VectorizedModuleInfo& vm_info;

    llvm::Function* createVectorFunction(llvm::Function* F, ps::VFABI& vfabi);

    struct GridMetadata {
        GridMetadata()
            : populated(false),
              omp_func(nullptr),
              gang_num(0),
              grid_size(nullptr) {}
        bool populated;
        VFABI vfabi;
        llvm::Function* omp_func;
        llvm::Value* gang_num;
        llvm::Value* grid_size;
        std::string subname;
    };

    std::unordered_map<llvm::Function*, VFABI> entry_points;

    void setGridGangNum(llvm::CallInst* inst, GridMetadata& launch_metadata);
    void setGridGangSize(llvm::CallInst* inst, GridMetadata& launch_metadata);
    void setGridSize(llvm::CallInst* inst, GridMetadata& launch_metadata);
    void setGridSubName(llvm::CallInst* inst, GridMetadata& launch_metadata);

    void setGridOmpFunction(llvm::CallInst* inst,
                            GridMetadata& launch_metadata);
    void finishGridMetadata(GridMetadata& launch_metadata);
    void findPsimCalls(
        std::unordered_map<llvm::CallInst*, GridMetadata>& launches,
        std::unordered_set<llvm::CallInst*>& insts_to_delete);
    void insertPsimGrids(
        std::unordered_map<llvm::CallInst*, GridMetadata>& launches);
    void findPSVEntryPoints();

    void preprocessFunction(llvm::Function* VF);
    void replaceUnreachableInsts(llvm::Function* F);
};

}  // namespace ps
