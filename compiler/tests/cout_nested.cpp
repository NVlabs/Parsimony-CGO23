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
#include <iostream>

#define GANG_SIZE 4

#pragma omp declare simd simdlen(GANG_SIZE) linear(i : 1)
inline __attribute__((always_inline)) void  bar(int i) {
    std::cout << "Lane " << i << " says hello!\n";
    psim_gang_sync();
}

int main() {
#psim gang_size(GANG_SIZE)
    {
        int i = psim_get_lane_num();
        if (i > 0) {
            bar(i);
        }
    }

    printf("Expected:\n");
    for (unsigned i = 1; i < GANG_SIZE; i++) {
        printf("Lane %d says hello!\n", i);
    }
    printf("(possibly interleaved)\n");
}
