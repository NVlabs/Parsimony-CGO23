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

#define SIMD_WIDTH 4

void foo_body(int* a, int i) {
    for (int j = 0; j < 32; j++) {
        a[i] += i;
        if (a[i] > 5) {
            break;
        }
    }
}

int main() {
    int a[SIMD_WIDTH] = {};
    int b[SIMD_WIDTH] = {};
#psim gang_size(SIMD_WIDTH)
    {
        int i = psim_get_lane_num();
        foo_body(a, i);
    }
    for (int i = 0; i < SIMD_WIDTH; i++) {
        foo_body(b, i);
    }
    for (int i = 0; i < SIMD_WIDTH; i++) {
        assert(a[i] == b[i]);
    }
    printf("Success!\n");
}
