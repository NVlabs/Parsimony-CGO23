/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#define SIMD_WIDTH 32

#include <math.h>
#include <parsim.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <cassert>
#include <iostream>

#define M 1

typedef struct __attribute__((packed)) {
    double x;
    float y;
    float z;
    float w;
    char c;
} S;

std::ostream& operator<<(std::ostream& os, const S& a) {
    os << a.x << " " << a.y << " " << a.z << " " << a.w << " " << (int)a.c;
    return os;
}

void init_val(S& a, int i) {
    a.x = i;
    a.y = i + 1000;
    a.z = i + 2;
    a.w = i + 42;
    a.c = i + 1;
}

bool operator==(const S& a, const S& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w && a.c == b.c;
}

int main(int argc, char* argv[]) {
    S* a = (S*)malloc(M * sizeof(S) * SIMD_WIDTH);
    S* b = (S*)malloc(M * sizeof(S) * SIMD_WIDTH);

    for (unsigned int i = 0; i < M * SIMD_WIDTH; i++) {
        init_val(a[i], i);
        memset(b, 0, sizeof(S) * M * SIMD_WIDTH);
    }
#psim gang_size(SIMD_WIDTH)
    {
        int i = psim_get_lane_num();
        PSIM_WARNINGS_OFF;
        b[i] = a[M * i];
        PSIM_WARNINGS_ON;
    }
    for (unsigned int i = 0; i < SIMD_WIDTH; i++) {
        assert(b[i] == a[M * i]);
    }

    printf("Success!\n");
    return 0;
}
