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

#define GANG_SIZE 4

struct S {
    int my_int;
    float my_float;
    double my_double;
};

#define VALS \
    { 1, 1.0, 2.0 }

int main() {
    uint64_t addrs[3][GANG_SIZE];
    S values[GANG_SIZE];
#psim gang_size(GANG_SIZE)
    {
        S s = VALS;
        size_t lane = psim_get_lane_num();
        addrs[0][lane] = (uint64_t)&s;
        addrs[1][lane] = (uint64_t)&s.my_int;
        addrs[2][lane] = (uint64_t)&s.my_float;

        values[lane].my_int = s.my_int + lane;
        values[lane].my_float = s.my_float + lane;
        values[lane].my_double = s.my_double + lane;
    }

    S ref = VALS;
    for (int i = 0; i < GANG_SIZE; i++) {
        assert(values[i].my_int == ref.my_int + i);
        assert(values[i].my_float == ref.my_float + i);
        assert(values[i].my_double == ref.my_double + i);
        // printf("%lx %lx %lx %ld %ld\n", addrs[0][i], addrs[1][i],
        //       addrs[1][i] - addrs[0][i], addrs[2][i],
        //       addrs[2][i] - addrs[1][i]);
        assert(addrs[0][i] == addrs[1][i]);
        assert(addrs[1][i] + sizeof(float) == addrs[2][i]);
    }
    printf("Success!\n");
}
