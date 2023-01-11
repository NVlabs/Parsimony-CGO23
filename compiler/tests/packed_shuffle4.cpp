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

int main() {
    int nelem = 54;
    uint8_t* a = (uint8_t*)malloc(nelem * sizeof(uint8_t*));
    uint8_t* b = (uint8_t*)malloc(nelem * sizeof(uint8_t*));
    uint8_t* ref = (uint8_t*)malloc(nelem * sizeof(uint8_t*));
    for (int i = 0; i < nelem; i++) {
        a[i] = i;
        b[i] = 0;
    }

    for (int i = 0; i < nelem; i++) {
        ref[i] = a[i % 3] + a[i / 3];
    }
#psim num_spmd_threads(nelem) gang_size(4 * 3)
    {
        size_t i = psim_get_thread_num();
        b[i] = a[i % 3] + a[i / 3];
    }
    bool error = false;
    for (int i = 0; i < nelem; i++) {
        if (b[i] != ref[i]) {
            printf("error @%d - %d != %d\n", i, b[i], ref[i]);
            error = true;
        }
    }
    assert(!error);
    printf("Success!\n");
}
