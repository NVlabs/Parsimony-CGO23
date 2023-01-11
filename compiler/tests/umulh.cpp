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
#include <cassert>

#define SIMD_WIDTH 96

inline uint16_t DivideBy255_sw(uint16_t value) {
    return (value + 1 + (value >> 8)) >> 8;
}

inline uint16_t DivideBy255_hw(uint16_t value) {
    return psim_umulh(value + 1, 257);
}

int main() {
    int a[SIMD_WIDTH] = {};
#psim gang_size(SIMD_WIDTH)
    {
        int i = psim_get_lane_num();
        a[i] = DivideBy255_hw(i * 1000);
    }

    for (int i = 0; i < SIMD_WIDTH; i++) {
        assert(a[i] == DivideBy255_sw(i * 1000));
    }
    printf("Success!\n");
}
