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

#include <unordered_map>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/Intrinsics.h>

#include <llvm/IR/IntrinsicsX86.h>

#include "vfabi.h"

namespace ps {

extern unsigned resolver_verbosity_level;

struct FunctionResolution {
    llvm::Function* function;
    VFABI vfabi;
};

typedef std::unordered_map<llvm::Function*, std::vector<FunctionResolution>>
    ResolverMap;

class FunctionResolver {
  public:
    FunctionResolver() {}

    FunctionResolution get(llvm::Function* f, VFABI& desired);
    void add(llvm::Function* f, FunctionResolution resolution);

    enum PsimApiEnum {
        GET_LANE_NUM,
        GET_GANG_NUM,
        GET_GANG_SIZE,
        GET_GRID_SIZE,
        GET_THREAD_NUM,
        GET_OMP_THREAD_NUM, //unused
        UADD_SAT,
        SADD_SAT,
        USUB_SAT,
        SSUB_SAT,
        UMULH,
        SHFL_SYNC,
        ZIP_SYNC,
        UNZIP_SYNC,
        GANG_SYNC,
        ATOMICADD_LOCAL,
        COLLECTIVE_ADD_ABS_DIFF,
        PSIM_API_NONE,
    };
    PsimApiEnum getPsimApiEnum(llvm::Function* f);

    std::unordered_map<PsimApiEnum, std::string> PsimApiEnumStrMap = {
        {GET_LANE_NUM, "psim_get_lane_num"},
        {GET_GANG_SIZE, "psim_get_gang_size"},
        {GET_GANG_NUM, "psim_get_gang_num"},
        {GET_GRID_SIZE, "psim_get_num_threads"},
        {GET_THREAD_NUM, "psim_get_thread_num"},
        {GET_OMP_THREAD_NUM, "omp_get_thread_num"},//unused
        {UADD_SAT, "psim_uadd_sat"},
        {SADD_SAT, "psim_sadd_sat"},
        {USUB_SAT, "psim_usub_sat"},
        {SSUB_SAT, "psim_ssub_sat"},
        {UMULH, "psim_umulh"},
        {COLLECTIVE_ADD_ABS_DIFF, "PsimCollectiveAddAbsDiff"},
        {SHFL_SYNC, "psim_shuffle_sync"},
        {ZIP_SYNC, "psim_zip_sync"},
        {GANG_SYNC, "psim_gang_sync"},
        {UNZIP_SYNC, "psim_unzip_sync"},
        {ATOMICADD_LOCAL, "psim_atomic_add_local"}};

    std::unordered_map<PsimApiEnum, llvm::Intrinsic::ID> LlvmInstrinsicMap =
        {{UADD_SAT, llvm::Intrinsic::uadd_sat},
         {SADD_SAT, llvm::Intrinsic::sadd_sat},
         {USUB_SAT, llvm::Intrinsic::usub_sat},
         {SSUB_SAT, llvm::Intrinsic::ssub_sat}};

    std::unordered_map<PsimApiEnum, llvm::Intrinsic::ID>
        Avx512InstrinsicMap = {
            {UMULH, llvm::Intrinsic::x86_avx512_pmulhu_w_512},
            {COLLECTIVE_ADD_ABS_DIFF, llvm::Intrinsic::x86_avx512_psad_bw_512}};

  private:
    ResolverMap resolver_map;

    FunctionResolution getBestVFABIMatch(
        std::vector<FunctionResolution>& resolutions, VFABI& desired);
};

}  // namespace ps
