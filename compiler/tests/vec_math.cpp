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
#include <cmath>

#define GANG_SIZE 32
int limit = 9;

bool compare_float(float x, float y, float epsilon = 1e-3) {
    if (fabs(x - y) < epsilon) {
        return true;
    }
    return false;
}

#define CHECK(i, a, b)                                                    \
    if (!compare_float((float)a, (float)b)) {                             \
        printf("Error @%d %s - %10.15f != %10.15f\n", i, #a #b, (float)a, \
               (float)b);                                                 \
        assert(0);                                                        \
    }

__attribute__((noinline)) void foo(float* a, float* b, float* c) {
#psim gang_size(GANG_SIZE)
    {
        int i = psim_get_lane_num();
        if (i < limit) {
            a[i] = expf(a[i]);
            b[i] = sin(i);
            c[i] = sqrt(i);
        }
    }
}

int main() {
    float a[GANG_SIZE] = {};
    float b[GANG_SIZE] = {};
    float c[GANG_SIZE] = {};
    for (int i = 0; i < GANG_SIZE; i++) {
        a[i] = i;
        b[i] = i;
        c[i] = i;
    }

    foo(a, b, c);

    for (int i = 0; i < GANG_SIZE; i++) {
        if (i < limit) {
            CHECK(i, a[i], (float)expf(i));
            CHECK(i, b[i], (float)sin(i));
            CHECK(i, c[i], (float)sqrt(i));
        } else {
            CHECK(i, a[i], i);
            CHECK(i, b[i], i);
            CHECK(i, c[i], i);
        }
    }
    printf("Success!\n");
}
