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

#include <cstddef>
#include <cstdint>

#define PSIM_WARNINGS_ON                                         \
    {                                                                \
        int __attribute__((annotate("warn_on"))) __psim_warn_on; \
        (void)__psim_warn_on;                                    \
    }

#define PSIM_WARNINGS_OFF                                          \
    {                                                                  \
        int __attribute__((annotate("warn_off"))) __psim_warn_off; \
        (void)__psim_warn_off;                                     \
    }

/* internal APIS used for code generation, not user-exposed */
extern "C" void __psim_set_grid_size(uint64_t grid_size) noexcept;
extern "C" void __psim_set_gang_num(uint64_t grid_num) noexcept;
extern "C" void __psim_set_gang_size(unsigned gang_size) noexcept;
extern "C" void __psim_set_grid_sub_name(const char* subname) noexcept;

extern "C" unsigned psim_get_lane_num() noexcept;
extern "C" uint64_t psim_get_gang_num() noexcept;
extern "C" unsigned psim_get_gang_size() noexcept;
extern "C" uint64_t psim_get_num_threads() noexcept;
extern "C" uint64_t psim_get_thread_num() noexcept;
extern "C" bool psim_is_tail_gang() noexcept;
extern "C" bool psim_is_head_gang() noexcept;

/* saturating signed and unsigned add/sub intrinsic */
template <typename T>
T psim_sadd_sat(T a, T b) noexcept;
template <typename T>
T psim_uadd_sat(T a, T b) noexcept;
template <typename T>
T psim_ssub_sat(T a, T b) noexcept;
template <typename T>
T psim_usub_sat(T a, T b) noexcept;

uint16_t psim_umulh(uint16_t a, uint16_t b) noexcept;

template <typename T>
struct PsimCollectiveAddAbsDiff {
    static constexpr const int N_ARCH = 8;
    typedef uint64_t vty
        __attribute__((vector_size(N_ARCH * sizeof(uint64_t))));
    vty var = {};

  public:
    template <typename T2>
    void AddAbsDiff(T2 a, T2 b) noexcept;

    T ReduceSum() {
        uint64_t sum = 0;
        for (int i = 0; i < N_ARCH; i++) {
            sum += var[i];
        }
        return sum;
    }
};

template <typename T1, typename T2>
T1 psim_shuffle_sync(T2 a, int src_lane) noexcept __attribute__((convergent));

template <typename T1, typename T2>
T1 psim_shuffle_sync(T2 a, T2 b, int src_index) noexcept
    __attribute__((convergent));

/* zip/unzip*/
template <typename T1, typename T2>
T1 psim_zip_sync(T2 a) noexcept __attribute__((convergent));

template <typename T1, typename T2>
T1 psim_unzip_sync(T2 a, uint32_t index) noexcept
    __attribute__((convergent));

void psim_gang_sync() noexcept __attribute__((convergent));

template <typename T1, typename T2>
void psim_atomic_add_local(T1* a, T2 value) noexcept;
