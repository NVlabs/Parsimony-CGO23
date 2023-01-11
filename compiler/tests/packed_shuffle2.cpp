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

#define GANG_SIZE 64

void foo(uint8_t* in, uint8_t* out) {
#psim gang_size(GANG_SIZE)
    {
        unsigned lane = psim_get_lane_num();

        uint8_t* ptr = &(in[3 * lane]);
        out[lane] = lane % 2 ? ptr[1] : ptr[2];
    }
}

int main() {
    uint8_t in[GANG_SIZE * 3];
    uint8_t out[GANG_SIZE] = {};
    for (int i = 0; i < GANG_SIZE * 3; i++) {
        in[i] = i;
    }
    foo(in, out);

    for (int i = 0; i < GANG_SIZE; i++) {
        if (i % 2) {
            assert(in[i * 3 + 1] == out[i]);
        } else {
            assert(in[i * 3 + 2] == out[i]);
        }
    }
    printf("Success!\n");
}
