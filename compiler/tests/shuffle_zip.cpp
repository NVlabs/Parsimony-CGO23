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

/* clang-format off */
#define PATTERN1 {0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1}
#define PATTERN2 {0x1, 0x0, 0x2, 0x1, 0x4, 0x3, 0x5, 0x4, 0x7, 0x6, 0x8, 0x7, 0xA, 0x9, 0xB, 0xA }

const int pattern[] = { 1,  0,  2,  1,  4,  3,  5,  4,  7,  6,  8,  7,  10, 9,  11, 10,
                        13, 12, 14, 13, 16, 15, 17, 16, 19, 18, 20, 19, 22, 21, 23, 22,
                        25, 24, 26, 25, 28, 27, 29, 28, 31, 30, 32, 31, 34, 33, 35, 34,
                        37, 36, 38, 37, 40, 39, 41, 40, 43, 42, 44, 43, 46, 45, 47, 46};
/* clang-format on */

int main() {
    uint8_t a[GANG_SIZE];
    uint8_t b[GANG_SIZE];
    uint8_t c[GANG_SIZE];

    for (int i = 0; i < GANG_SIZE; i++) {
        a[i] = i;
        b[i] = 0;
    }

    for (int i = 0; i < GANG_SIZE; i++) {
        c[i] = a[pattern[i]];
    }

#psim gang_size(GANG_SIZE)
    {
        uint8_t lane = psim_get_lane_num();
        uint8_t val = a[lane];
        const int shfl1[] = PATTERN1;
        const int shfl2[] = PATTERN2;

        uint32_t val32 = psim_zip_sync<uint32_t>(val);
        uint32_t permute32 =
            psim_shuffle_sync<uint32_t>(val32, shfl1[lane % 16]);
        uint8_t permute8 = psim_unzip_sync<uint8_t>(permute32, 0);
        uint8_t shuffle8 = psim_shuffle_sync<uint8_t>(
            permute8, shfl2[lane % 16] + lane / 16 * 16);
        b[lane] = shuffle8;
#if 0
        printf(
            "lane %4d - val 0x%02x - val32 0x%08x - shfl1 %4d - permute32 "
            "0x%08x - "
            "permute8 0x%02x - shfl2 %4d - shuffle8 0x%02x - expected 0x%02x\n",
            lane, val, val32, shfl1[lane % 16], permute32, permute8,
            shfl2[lane], shuffle8, c[lane]);
#endif
    }

    bool error = false;
    for (int i = 0; i < GANG_SIZE; i++) {
        if (b[i] != c[i]) {
            printf("Error @%d - %d != %d\n", i, b[i], c[i]);
            error = true;
        }
    }
    if (!error) {
        printf("Success!\n");
    } else {
        printf("Fail!\n");
    }
}
