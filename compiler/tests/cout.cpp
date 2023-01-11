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
#include <iostream>

int main() {
#psim gang_size(4)
    {
        int i = psim_get_lane_num();

        // This may print "Lane Lane 0 1 says hello!\nsays hello!\n", and that's
        // OK!
        std::cout << "Lane " << i << " says hello!\n";
    }

    printf("Expected:\n");
    for (unsigned i = 0; i < 4; i++) {
        printf("Lane %d says hello!\n", i);
    }
    printf("(possibly interleaved)\n");
}
