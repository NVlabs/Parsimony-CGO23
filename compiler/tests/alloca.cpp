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

#define GANG_SIZE 4

int main() {
    int a[12][GANG_SIZE] = {};
#psim gang_size(GANG_SIZE)
    {
        uint8_t lane = psim_get_lane_num();
        PSIM_WARNINGS_OFF
        uint8_t data[12] = {};
        for (int i = 0; i < 12; i++) {
            data[i] += lane * i;
        }
        for (int i = 0; i < 12; i++) {
            a[i][lane] = data[i];
        }
        PSIM_WARNINGS_ON
    }
    uint64_t sum = 0;
    uint64_t ref_sum = 0;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < GANG_SIZE; j++) {
            sum += a[i][j];
            ref_sum += i * j;
        }
    }
    assert(sum == ref_sum);
    printf("Success!\n");
}
