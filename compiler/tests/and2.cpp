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

#define GANG_SIZE 16

__attribute__((noinline)) void foo(uint32_t* a) {
#psim gang_size(GANG_SIZE)
    {
        size_t t = psim_get_thread_num();
        size_t pos = (t / 4) * 2;
        a[pos] = t;
    }
}

int main() {
    uint32_t a[GANG_SIZE * 2] = {};
    uint32_t res[GANG_SIZE * 2] = {};
    for (int i = 0; i < GANG_SIZE * 2; i++) {
        a[i] = 0;
        if (i < GANG_SIZE) {
            res[(i / 4) * 2] = i;
        }
    }

    foo(a);

    for (int i = 0; i < GANG_SIZE * 2; i++) {
        assert(a[i] == res[i]);
    }
    printf("Success!\n");
}
