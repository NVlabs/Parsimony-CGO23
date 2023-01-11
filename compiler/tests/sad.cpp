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

int main() {
    int array_size = 12345;
    uint8_t* a = (uint8_t*)malloc(array_size * sizeof(uint8_t));
    uint8_t* b = (uint8_t*)malloc(array_size * sizeof(uint8_t));

    uint64_t ref_sad = 0;
    for (int i = 0; i < array_size; i++) {
        a[i] = rand() % 256;
        b[i] = rand() % 256;
        ref_sad += std::max(a[i], b[i]) - std::min(a[i], b[i]);
    }

    PsimCollectiveAddAbsDiff<uint64_t> _sum;

#psim num_spmd_threads(array_size) gang_size(GANG_SIZE)
    {
        size_t i = psim_get_thread_num();

        _sum.AddAbsDiff(a[i], b[i]);
    }
    uint64_t sum = _sum.ReduceSum();
    assert(sum == ref_sad);
    printf("Success!\n");
}
