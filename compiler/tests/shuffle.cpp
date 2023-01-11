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

STATIC_INLINE int RestrictRange(int value, int min = 0, int max = 255) {
    return std::max(min, std::min(max, value));
}

void serial(uint8_t* dst, uint8_t* src) {
    for (int i = 0; i < GANG_SIZE; i += 4) {
        float alpha = src[3] ? 255.00001f / src[3] : 0.0f;
        dst[0] = RestrictRange(int(src[0] * alpha));
        dst[1] = RestrictRange(int(src[1] * alpha));
        dst[2] = RestrictRange(int(src[2] * alpha));
        dst[3] = src[3];

        dst += 4;
        src += 4;
    }
}

int main() {
    uint8_t a[GANG_SIZE] = {};
    uint8_t b[GANG_SIZE] = {};
    uint8_t c[GANG_SIZE] = {};
    for (int i = 0; i < GANG_SIZE; i++) {
        a[i] = i;
        b[i] = 0;
        c[i] = 0;
    }

#psim gang_size(GANG_SIZE)
    {
        uint32_t lane = psim_get_lane_num();
        uint8_t val = a[lane];

        uint32_t shf_ptrn1 = (lane % 16) * 4 + lane / 16;
        uint32_t color = psim_shuffle_sync<uint32_t>(val, shf_ptrn1);

        float alpha = 0;
        if (lane >= 48) {
            alpha = color ? 255.00001f / color : 0.0f;
        }

        uint32_t shf_ptrn2 = 48 + lane % 16;
        alpha = psim_shuffle_sync<float>(alpha, shf_ptrn2);

        uint32_t out32 = color;
        if (lane < 48) {
            out32 = RestrictRange(int(color * alpha));
        }
        uint32_t shf_ptrn3 = (lane % 4) * 16 + lane / 4;
        uint8_t out = psim_shuffle_sync<uint8_t>(out32, shf_ptrn3);

        b[lane] = out;
    }

    serial(c, a);

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
        assert(0);
    }
}
