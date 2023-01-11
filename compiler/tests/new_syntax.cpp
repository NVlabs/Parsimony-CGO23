/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <omp.h>
#include <parsim.h>
#include <cstdio>
#include <cstdlib>

struct dim3 {
    int x, y, z;
};
#define NELEM 123

int main() {
    dim3 a[NELEM];
    int b[NELEM];

    for (int i = 0; i < NELEM; i++) {
        a[i].x = i;
        a[i].y = i * 2;
        a[i].z = i * 3;
    }

#psim parallel num_spmd_threads(NELEM) gang_size(32)
    {
        int i = psim_get_lane_num() +
                psim_get_gang_num() * psim_get_gang_size();
        b[i] = a[i].x * 2 + psim_get_num_threads();
    }

    for (int i = 0; i < NELEM; i++) {
        if (b[i] != i * 2 + NELEM) {
            printf("Fail! b[%d] = %d, expected %d\n", i, b[i],
                   i * 2 + NELEM);
            exit(2);
        }
    }

    printf("Success!\n");
    return 0;
}
