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
#include <algorithm>
#include <cassert>

#define GANG_SIZE 128
#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

int main() {
    int array_size = 343;
    int backgroundStride = 37;
    uint8_t* a = (uint8_t*)malloc(array_size * sizeof(uint8_t));
    uint8_t* b =
        (uint8_t*)malloc((array_size + backgroundStride) * sizeof(uint8_t));

    uint64_t ref_sad = 0;
    for (int i = 0; i < array_size; i++) {
        a[i] = rand() % 256;
    }
    for (int i = 0; i < array_size + backgroundStride; i++) {
        b[i] = rand() % 256;
    }

    for (int i = 0; i < array_size; i++) {
        uint8_t v0 = a[i];
        uint8_t v1 = b[i];
        ref_sad += std::max(v0, v1) - std::min(v0, v1);
    }

    PsimCollectiveAddAbsDiff<uint64_t> _sum = {};

    b += backgroundStride + 1;

#psim num_spmd_threads(array_size) gang_size(GANG_SIZE)
    {
        size_t col = psim_get_thread_num();
        uint8_t v0 = a[col];
        uint8_t v1 = b[col - backgroundStride - 1];
        _sum.AddAbsDiff(v0, v1);
    }

    uint64_t sum = _sum.ReduceSum();
    assert(sum == ref_sad);
    printf("Success!\n");
}
