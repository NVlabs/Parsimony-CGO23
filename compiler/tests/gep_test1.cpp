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
#include <string.h>
#include <algorithm>
#include <cassert>

#define GANG_SIZE 64

__attribute__((noinline)) void foo(uint8_t in[3][64], uint8_t out[3][64]) {
#psim gang_size(GANG_SIZE)
    {
        uint8_t lane = psim_get_lane_num();
        uint8_t val[3];
#pragma unroll
        for (int i = 0; i < 3; i++) {
            int idx = i * 64 + lane;
            val[i] = in[idx % 3][idx / 3];
            //printf("%3d ", val[i]);
            psim_gang_sync();
            //if (lane == 0) printf("\n");
        }

#pragma unroll
        for (int i = 0; i < 3; i++) {
            out[i][lane] = val[i];
        }
    }
}

int main() {
    assert(GANG_SIZE == 64);
    uint8_t in[3][GANG_SIZE] = {};
    uint8_t out[3][GANG_SIZE] = {};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < GANG_SIZE; j++) {
            in[i][j] = 64 * i + j;
            out[i][j] = 0;
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < GANG_SIZE; j++) {
            int idx = i * 64 + j;
            uint8_t ref = in[idx % 3][idx / 3];
            //printf("%3d ", ref);
        }
        //printf("\n");
    }
    //printf("\n");

    foo(in, out);

    bool success = true;
    int max_err_count = 10;
    int err_count = max_err_count;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < GANG_SIZE; j++) {
            int idx = i * 64 + j;
            uint8_t ref = in[idx % 3][idx / 3];
            if (out[i][j] != ref) {
                if (err_count-- > 0) {
                    printf(
                        "Error (max %d) - @%d,%d - %d != "
                        "%d\n",
                        max_err_count, i, j, out[i][j], ref);
                }
                success = false;
            }
        }
    }
    assert(success);
    printf("Success!\n");
}
