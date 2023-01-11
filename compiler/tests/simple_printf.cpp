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
#include <string>

#define GANG_SIZE 32

void __attribute__((always_inline)) foo_body(int i, char* buf) {
    if (i < 7) {
        sprintf(buf, "%d ", i);
    }
}

int main() {
    char buffs[GANG_SIZE][1024] = {};
#psim gang_size(GANG_SIZE)
    {
        int i = psim_get_lane_num();
        foo_body(i, buffs[i]);
    }
    for (int i = 0; i < GANG_SIZE; i++) {
        char buf[1024] = {};
        foo_body(i, buf);
        assert(std::string(buf) == std::string(buffs[i]));
    }
    printf("Success!\n");
}
