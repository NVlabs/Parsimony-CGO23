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
#include <algorithm>
#include <cassert>

#define GANG_SIZE 64

int main() {
    uint8_t a[GANG_SIZE] = {};
    uint32_t b[GANG_SIZE] = {};
    uint32_t c[GANG_SIZE] = {};
    uint8_t d[GANG_SIZE] = {};

    for (int i = 0; i < GANG_SIZE/4; i++) {
        c[i] = 0;
        for(int j = 3 ; j >= 0; j--){
            c[i] |= ((4*(uint8_t)i) + (uint8_t)j) << j*8;
        }
    }

    for (int i = 0; i < GANG_SIZE; i++) {
        a[i] = i;
        b[i] = 0;
        d[i] = 0;
        c[i] = c[i % 16];
    }

#psim gang_size(GANG_SIZE)
    {
        uint32_t lane = psim_get_lane_num();
        b[lane] =  psim_zip_sync<uint32_t>(a[lane]);
        d[lane] = psim_unzip_sync<uint8_t>(b[lane], 1);
    }


    bool error = false;
    for (int i = 0; i < GANG_SIZE; i++) {
        if (b[i] != c[i]) {
            printf("Error1 @%d - 0x%x != 0x%x \n", i, b[i], c[i]);
            error = true;
        }
    }
    for (int i = 0; i < GANG_SIZE; i++) {
        if (a[i] != d[i]) {
            printf("Error2 @%d - %d != %d\n", i, a[i], d[i]);
            error = true;
        }
    }

    if (!error) {
        printf("Success!\n");
    } else {
        printf("Fail!\n");
    }
}
