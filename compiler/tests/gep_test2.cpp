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
#include <string>

#define GANG_SIZE 64

typedef struct __attribute__((packed)) my_struct_t {
    uint8_t a[3];

    bool operator==(const my_struct_t& o) const {
        for (int i = 0; i < 3; i++) {
            if (a[i] != o.a[i]) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const my_struct_t& o) const { return !(*this == o); }

    std::string toString() {
        std::string s = "(";
        for (int i = 0; i < 3; i++) {
            s += std::to_string(a[i]) + ",";
        }
        s += ")";
        return s;
    }
} my_struct_t;

__attribute__((noinline)) void foo(my_struct_t* in, my_struct_t* out) {
#psim gang_size(GANG_SIZE)
    {
        // for (int lane = 0; lane < GANG_SIZE; lane++) {
        uint8_t lane = psim_get_lane_num();
        uint8_t val = in[lane].a[2];

        val++;
        out[lane].a[2] = val;

        // printf("in: %s -- out: %s\n", in[lane].toString().c_str(),
        //       out[lane].toString().c_str());
    }
}

int main() {
    my_struct_t in[GANG_SIZE];
    my_struct_t out[GANG_SIZE];
    my_struct_t ref[GANG_SIZE];
    for (int i = 0; i < GANG_SIZE; i++) {
        for (int j = 0; j < 3; j++) {
            in[i].a[j] = i * 3 + j;
        }
        ref[i] = in[i];
        out[i] = in[i];
        ref[i].a[2]++;
        //printf("in: %s -- ref: %s\n", in[i].toString().c_str(),
        //       ref[i].toString().c_str());
    }

    foo(in, out);

    bool success = true;
    int max_err_count = 10;
    int err_count = max_err_count;
    for (int i = 0; i < GANG_SIZE; i++) {
        if (out[i] != ref[i]) {
            if (err_count-- > 0) {
                printf("Error (max %d) - @%d - %s != %s\n", max_err_count, i,
                       out[i].toString().c_str(), ref[i].toString().c_str());
            }
            success = false;
        }
    }
    assert(success);
    printf("Success!\n");
}
