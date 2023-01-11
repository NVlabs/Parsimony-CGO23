/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <immintrin.h>
#include <parsim.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <algorithm>
#include <cassert>

#include "Simd/SimdConversion.h"

/* return time in microseconds */
static __attribute__((unused)) double GetTimer() {
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return ((double)1000000 * (double)tv.tv_sec + (double)tv.tv_usec);
}

#define GANG_SIZE 64

__attribute__((noinline)) void interleave3_simd(uint8_t in[3][64],
                                                uint8_t out[3][64]) {
    __m512i _in[3], _out[3];

#pragma unroll
    for (int i = 0; i < 3; i++) {
        memcpy(&_in[i], in[i], 64 * sizeof(uint8_t));
    }
    _out[0] = Simd::Avx512bw::InterleaveBgr<0>(_in[0], _in[1], _in[2]);
    _out[1] = Simd::Avx512bw::InterleaveBgr<1>(_in[0], _in[1], _in[2]);
    _out[2] = Simd::Avx512bw::InterleaveBgr<2>(_in[0], _in[1], _in[2]);

#pragma unroll
    for (int i = 0; i < 3; i++) {
        memcpy(out[i], &_out[i], 64 * sizeof(uint8_t));
    }
}

__attribute__((noinline)) void interleave3_v1(uint8_t in[3][64],
                                              uint8_t out[3][64]) {
#psim gang_size(GANG_SIZE)
    {
        uint8_t lane = psim_get_lane_num();
#pragma unroll
        for (int j = 0; j < 3; j++) {
            int idx = (j * 64 + lane) / 3;
#pragma unroll
            for (int i = 0; i < 3; i++) {
                int idx2 = (lane % 3) == ((i + 2 * j) % 3) ? idx : -1;
                out[j][lane] |= psim_shuffle_sync<uint8_t>(in[i][lane], idx2);
            }
        }
    }
}

__attribute__((noinline)) void interleave3_v2(uint8_t in[3][64],
                                              uint8_t out[3][64]) {
#psim gang_size(GANG_SIZE)
    {
        uint8_t lane = psim_get_lane_num();

        uint32_t col32[3];
        uint8_t col8s[3][3];

#pragma unroll
        for (int i = 0; i < 3; i++) {
            col32[i] = psim_zip_sync<uint32_t>(in[i][lane]);
        }
        int shfl_permute32 = lane % 4 + (lane / 12) * 4;
        int shfl_permute8[3][3] = {};

#pragma unroll
        for (int i = 0; i < 3; i++) {
            uint32_t col32s =
                psim_shuffle_sync<uint32_t>(col32[i], shfl_permute32);
#pragma unroll
            for (int j = 0; j < 3; j++) {
                col8s[i][j] = psim_unzip_sync<uint8_t>(col32s, j);

                int l = lane + 64 * i;
                shfl_permute8[i][j] =
                    (l % 3 == j) ? ((l / 3) % 16) + ((lane / 16) * 16) : -1;
            }
        }

#pragma unroll
        for (int i = 0; i < 3; i++) {
            out[i][lane] =
                psim_shuffle_sync<uint8_t>(col8s[0][i], shfl_permute8[i][0]) |
                psim_shuffle_sync<uint8_t>(col8s[1][i], shfl_permute8[i][1]) |
                psim_shuffle_sync<uint8_t>(col8s[2][i], shfl_permute8[i][2]);
        }
    }
}

__attribute__((noinline)) void interleave3_v3(uint8_t in[3][64],
                                              uint8_t out[3][64]) {
#if 0
#psim gang_size(GANG_SIZE)
    {
        uint8_t lane = psim_get_lane_num();
        uint8_t rgb[3];
#pragma unroll
        for (int i = 0; i < 3; i++) {
            int idx = i * 64 + lane;
            rgb[i] = in[idx % 3][idx / 3];
        }

#pragma unroll
        for (int i = 0; i < 3; i++) {
            out[i][lane] = rgb[i];
        }
    }
#else
#psim gang_size(192)
    {
        uint8_t lane = psim_get_lane_num();
        uint8_t val = in[lane % 3][lane / 3];
        out[lane / 64][lane % 64] = val;
    }
#endif
}

int main() {
    assert(GANG_SIZE == 64);
    uint8_t in[3][GANG_SIZE] = {};
    uint8_t out[3][GANG_SIZE] = {};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < GANG_SIZE; j++) {
            in[i][j] = 64 * i + j;
            out[i][j] = 0;
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < GANG_SIZE; j++) {
            int idx = i * 64 + j;
            uint8_t ref = in[idx % 3][idx / 3];
            // printf("%3d ", ref);
        }
        // printf("\n");
    }
    // printf("\n");

    int versions = 4;
    const char* version[] = {"simd", "psv_v1", "psv_v2", "psv_v3"};
    bool success = true;
    int reps = 100000;
    for (int v = 0; v < versions; v++) {
        double t0 = GetTimer();
        if (v == 1) {
            for (int rep = 0; rep < reps; rep++) {
                interleave3_v1(in, out);
            }
        } else if (v == 2) {
            for (int rep = 0; rep < reps; rep++) {
                interleave3_v2(in, out);
            }
        } else if (v == 3) {
            for (int rep = 0; rep < reps; rep++) {
                interleave3_v3(in, out);
            }
        } else if (v == 0) {
            for (int rep = 0; rep < reps; rep++) {
                interleave3_simd(in, out);
            }
        }
        double time = GetTimer() - t0;

        int max_err_count = 10;
        int err_count = max_err_count;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < GANG_SIZE; j++) {
                int idx = i * 64 + j;
                uint8_t ref = in[idx % 3][idx / 3];
                if (out[i][j] != ref) {
                    if (err_count-- > 0) {
                        printf(
                            "Error (max %d) for version %s - @%d,%d - %d != "
                            "%d\n",
                            max_err_count, version[v], i, j, out[i][j], ref);
                    }
                    success = false;
                }
                out[i][j] = 0;
            }
        }
        printf("time version %s: %f\n", version[v], time);
    }
    assert(success);
    printf("Success!\n");
}
