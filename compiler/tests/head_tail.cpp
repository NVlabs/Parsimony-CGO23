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

const int DELTA[12] = {-1, 0, 1, -2, 0, 2, -2, 0, +2, -1, 0, 1};

void foo(uint8_t* a, uint8_t* b, size_t len) {
    size_t num_gangs = CEILDIV(len, GS);
#psim num_spmd_gangs(num_gangs) gang_size(GS)
    {
        size_t col = psim_get_thread_num();
        size_t lane = psim_get_lane_num();

        uint8_t val = 0;
#pragma unroll
        for (int i = 0; i < 12; i++) {
            int fold = 0;
            int64_t sum = (int64_t)lane + DELTA[i];
            if (psim_is_head_gang()) {
                if (sum < 0) {
                    fold = 2;
                }
            } else if (psim_is_tail_gang()) {
                col = len - psim_get_gang_size() + lane;
                if (sum >= psim_get_gang_size()) {
                    fold = -2;
                }
            }

            size_t index = col + DELTA[i] + fold;
            val += a[index];
        }
        b[col] = val;
    }
}

int main() {
    size_t len = 141;

    uint8_t* a = (uint8_t*)malloc(len * sizeof(uint8_t));
    uint8_t* b = (uint8_t*)malloc(len * sizeof(uint8_t));

    for (int i = 0; i < len; i++) {
        a[i] = i;
        b[i] = 0;
    }
    foo(a, b, len);

    bool success = true;
    for (size_t l = 0; l < len; l++) {
        uint8_t val = 0;
        for (int i = 0; i < 12; i++) {
            int fold = 0;
            int64_t sum = (int64_t)l + DELTA[i];
            if (sum < 0) {
                fold = 2;
            } else if (sum >= len) {
                fold = -2;
            }
            size_t index = l + DELTA[i] + fold;
            assert(index < len);
            val += a[index];
        }

        if (b[l] != val) {
            printf("Error@%ld: %d != %d\n", l, b[l], val);
            success = false;
        }
    }
    assert(success);
    printf("Success!\n");
}
