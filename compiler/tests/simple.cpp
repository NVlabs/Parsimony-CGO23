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

#define GANG_SIZE 32
int limit = 9;

int main() {
    int a[GANG_SIZE] = {};
#psim gang_size(GANG_SIZE)
    {
        int i = psim_get_lane_num();
        if (i < limit) {
            a[i] = i;
        }
    }

    for (int i = 0; i < GANG_SIZE; i++) {
        if (i < limit) {
            assert(a[i] == i);
        } else {
            assert(a[i] == 0);
        }
    }
    printf("Success!\n");
}
