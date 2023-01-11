/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <parsim.h>
#include <stdio.h>
#include <algorithm>
#include <cassert>

#define SIMD_WIDTH 64

#define STATIC_INLINE static inline __attribute__((always_inline))

int main() {
    uint8_t a[SIMD_WIDTH] = {};
    uint32_t b[SIMD_WIDTH] = {};
    uint32_t c[SIMD_WIDTH] = {};
    for (int i = 0; i < SIMD_WIDTH; i++) {
        a[i] = i;
        b[i] = 0;
        c[i] = (i % 16) * 4 + (i / 16);
    }

#psim gang_size(SIMD_WIDTH)
    {
        uint32_t lane = psim_get_lane_num();
        int shfl_idx = (lane % 16) * 4 + (lane / 16);
        b[lane] = psim_shuffle_sync<uint32_t>(a[lane], shfl_idx);
    }

    bool error = false;
    for (int i = 0; i < SIMD_WIDTH; i++) {
        if (b[i] != c[i]) {
            printf("Error @%d - %d != %d\n", i, b[i], c[i]);
            error = true;
        }
    }
    if (!error) {
        printf("Success!\n");
    } else {
        printf("Fail!\n");
    }
}
