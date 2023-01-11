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
#include <stdlib.h>
#include <cassert>

#define CEILDIV(a, b) (((a) + ((b)-1)) / (b))

#define GS 64

void foo(uint8_t* a, size_t size) {
    size_t num_gangs = CEILDIV(size, GS);
#psim num_spmd_gangs(num_gangs) gang_size(GS)
    {
        size_t col = psim_get_thread_num();
        size_t lane = psim_get_lane_num();
        if (psim_get_gang_num() == num_gangs - 1) {
            col = size - GS + lane;
        }
        a[col] = col;
    }
}

int main() {
    size_t size = 141;

    uint8_t* a = (uint8_t*)malloc(size * sizeof(uint8_t));

    for (int i = 0; i < size; i++) {
        a[i] = 0;
    }
    foo(a, size);
    for (int i = 0; i < size; i++) {
        assert(a[i] == i);
    }
    printf("Success!\n");
}
