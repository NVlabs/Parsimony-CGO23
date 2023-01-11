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
#include <cassert>

#define GANG_SIZE 32

int permute(int l) {
    if (l == GANG_SIZE - 1) {
        l = 0;
    } else if (l % 8 == 0) {
        l = l % 4;
    }
    return l;
}

int main() {
    int nelem = GANG_SIZE;
    int* a = (int*)malloc(nelem * sizeof(int));
    int* b = (int*)malloc(nelem * sizeof(int));
    for (int i = 0; i < nelem; i++) {
        a[i] = i + 100000;
        b[i] = 0;
    }

#psim num_spmd_threads(nelem) gang_size(GANG_SIZE)
    {
        int l = psim_get_lane_num();
        size_t w = psim_get_gang_num();
        size_t c = psim_get_thread_num();

        l = permute(l);
        b[c] = a[w * GANG_SIZE + l];
    }

    for (int i = 0; i < nelem; i++) {
        assert(a[permute(i)] == b[i]);
    }
    printf("Success!\n");
}
