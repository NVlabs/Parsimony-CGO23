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
#include <cassert>
#include <cstdio>

#pragma omp declare simd simdlen(32)
static int __attribute__((noinline)) foo(int a) { return a + 1; }

int main() {
    int a[32];
    for (int i = 0; i < 32; i++) {
        a[i] = i;
    }

#psim gang_size(32)
    {
        int i = psim_get_lane_num();
        a[i] = foo(a[i]) + 2;
    }

    for (int i = 0; i < 32; i++) {
        assert(a[i] == i + 3);
    }

    printf("Success!\n");
    return 0;
}
