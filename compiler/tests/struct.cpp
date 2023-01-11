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

#define GANG_SIZE 32

struct S {
    float* f;
    int i[4];
};

int main() {
    S s[GANG_SIZE];
    int r[GANG_SIZE];

    for (int i = 0; i < GANG_SIZE; i++) {
        s[i].f = (float*)malloc(sizeof(float));
        *s[i].f = (float)i;
        for (int j = 0; j < 4; j++) {
            s[i].i[j] = i + j;
        }
    }

#psim gang_size(GANG_SIZE)
    {
        int i = psim_get_lane_num();
        r[i] = s[i].i[2];
    }

    for (int i = 0; i < GANG_SIZE; i++) {
        if (r[i] != i + 2) {
            printf("Error: r[%d] == %d, expected %d\n", i, r[i], i + 2);
            exit(2);
        }
    }
    printf("Success!\n");
}
