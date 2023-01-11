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

#define SIMD_WIDTH 32
int limit = 9;

int main() {
    int a[SIMD_WIDTH + 1] = {};
    int b[SIMD_WIDTH + 1] = {};
#psim gang_size(SIMD_WIDTH)
    {
        int i = psim_get_lane_num();  // stride 1
        a[i] = i;                         // packed

        if (i > limit) {
            i += 1;  // i becomes varying
        }
        PSIM_WARNINGS_OFF;
        b[i] = i;  // scatter
        PSIM_WARNINGS_ON;
    }

    for (int i = 0; i < SIMD_WIDTH; i++) {
        assert(a[i] == i);

        if (i > limit) {
            assert(b[i + 1] == i + 1);
        } else {
            assert(b[i] == i);
        }
    }
    printf("Success!\n");
}
