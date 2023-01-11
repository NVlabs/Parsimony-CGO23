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
const int table1[4] = {0, 1, 2, 3};
const int table2[4][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}};

int main() {
    int a[GANG_SIZE] = {};
    int b[GANG_SIZE] = {};
#psim gang_size(GANG_SIZE)
    {
        int i = psim_get_lane_num();
        int val = table1[i % 4];
        a[i] = psim_shuffle_sync<int>(i, val);

        int val2 = table2[i % 4][(i + 1) % 4];
        b[i] = psim_shuffle_sync<int>(i, val2);
    }
    for (int i = 0; i < GANG_SIZE; i++) {
        int val = table1[i % 4];
        int val2 = table2[i % 4][(i + 1) % 4];

        assert(a[i] == val);
        assert(b[i] == val2);
    }
    printf("Success!\n");
}
