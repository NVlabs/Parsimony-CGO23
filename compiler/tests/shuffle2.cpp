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

#define STATIC_INLINE static inline __attribute__((always_inline))

STATIC_INLINE uint32_t gen_shuffle_ptrn(int lane) {
    uint32_t shuffle_ptrn = 0;
    if (lane > 9) {
        shuffle_ptrn = ((lane % 16) * 4 + lane / 16 + 4) % GANG_SIZE;
    } else {
        shuffle_ptrn = ((lane % 16) * 4 + lane / 16 * 9) % GANG_SIZE;
    }
    return shuffle_ptrn;
}

int main() {
    uint8_t a[GANG_SIZE] = {};
    uint8_t b[GANG_SIZE] = {};
    for (int i = 0; i < GANG_SIZE; i++) {
        a[i] = i;
        b[i] = 0;
    }

#psim gang_size(GANG_SIZE)
    {
        uint32_t lane = psim_get_lane_num();
        uint8_t val = a[lane];

        uint32_t shuffle_ptrn = gen_shuffle_ptrn(lane);
        b[lane] = psim_shuffle_sync<uint32_t>(val, shuffle_ptrn);
    }

    bool error = false;
    for (int i = 0; i < GANG_SIZE; i++) {
        if (b[i] != a[gen_shuffle_ptrn(i)]) {
            printf("Error @%d - %d != %d\n", i, b[i], a[gen_shuffle_ptrn(i)]);
            error = true;
        }
    }
    if (!error) {
        printf("Success!\n");
    } else {
        printf("Fail!\n");
        assert(0);
    }
}
