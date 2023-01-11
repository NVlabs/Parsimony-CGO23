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
#include <cstdlib>

#define GANG_SIZE 32
int limit = 9;
const int stride = 4;

int main() {
    int* a = (int*)malloc(GANG_SIZE * stride * sizeof(int));
    for (int i = 0; i < GANG_SIZE * stride; i++) {
        a[i] = 0;
    }
#psim gang_size(GANG_SIZE)
    {
        int i = psim_get_lane_num();
        if (i < limit) {
            a[i * stride] = i;
        }
    }

    for (int i = 0; i < GANG_SIZE; i++) {
        if (i < limit) {
            assert(a[i * stride] == i);
        } else {
            assert(a[i * stride] == 0);
        }
    }
    printf("Success!\n");
}
