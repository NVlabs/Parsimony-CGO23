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

#define GANG_SIZE 64

/* return time in microseconds */
static __attribute__((unused)) double GetTimer() {
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return ((double)1000000 * (double)tv.tv_sec + (double)tv.tv_usec);
}

#define SIMD_INT_AS_LONGLONG(a) (((long long)a) & 0xFFFFFFFF)

#define SIMD_LL_SET2_EPI32(a, b) \
    SIMD_INT_AS_LONGLONG(a) | (SIMD_INT_AS_LONGLONG(b) << 32)

#define SIMD_MM512_SETR_EPI32(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, \
                              ac, ad, ae, af)                                 \
    {                                                                         \
        SIMD_LL_SET2_EPI32(a0, a1), SIMD_LL_SET2_EPI32(a2, a3),               \
            SIMD_LL_SET2_EPI32(a4, a5), SIMD_LL_SET2_EPI32(a6, a7),           \
            SIMD_LL_SET2_EPI32(a8, a9), SIMD_LL_SET2_EPI32(aa, ab),           \
            SIMD_LL_SET2_EPI32(ac, ad), SIMD_LL_SET2_EPI32(ae, af)            \
    }

const __m512i K32_PERMUTE_FOR_TWO_UNPACK =
    SIMD_MM512_SETR_EPI32(0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA,
                          0xE, 0x3, 0x7, 0xB, 0xF);

__attribute__((noinline)) void interleave4_simd(uint8_t in[4][64],
                                                uint8_t out[4][64]) {
    __m512i _in[4], _out[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        memcpy(&_in[i], in[i], 64 * sizeof(uint8_t));
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
        _in[i] = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _in[i]);
    }

    __m512i mix[4];

    mix[0] = _mm512_unpacklo_epi8(_in[0], _in[1]);
    mix[1] = _mm512_unpackhi_epi8(_in[0], _in[1]);
    mix[2] = _mm512_unpacklo_epi8(_in[2], _in[3]);
    mix[3] = _mm512_unpackhi_epi8(_in[2], _in[3]);

    _out[0] = _mm512_unpacklo_epi16(mix[0], mix[2]);
    _out[1] = _mm512_unpackhi_epi16(mix[0], mix[2]);
    _out[2] = _mm512_unpacklo_epi16(mix[1], mix[3]);
    _out[3] = _mm512_unpackhi_epi16(mix[1], mix[3]);

#pragma unroll
    for (int i = 0; i < 4; i++) {
        memcpy(out[i], &_out[i], 64 * sizeof(uint8_t));
    }
}

__attribute__((noinline)) void interleave4_v1(uint8_t in[4][64],
                                              uint8_t out[4][64]) {
#psim gang_size(GANG_SIZE)
    {
        uint8_t lane = psim_get_lane_num();
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int idx = (lane / 4) + j * 16;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                out[j][lane] |= psim_shuffle_sync<uint8_t>(
                    in[i][lane], lane % 4 == i ? idx : -1);
            }
        }
    }
}

__attribute__((noinline)) void interleave4_v2(uint8_t in[4][64],
                                              uint8_t out[4][64]) {
#psim gang_size(GANG_SIZE)
    {
        uint8_t lane = psim_get_lane_num();
        int shfl32_ptrn = (lane * 4) % 16 + (lane / 4);

        int shfl8_lo = lane / 2 + (lane % 2) * 64 + lane / 16 * 8;
        int shfl8_hi = 8 + shfl8_lo;
        int shfl16_lo = lane / 2 + (lane % 2) * 64 + lane / 8 * 4;
        int shfl16_hi = 4 + shfl16_lo;

        uint8_t _in[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            uint32_t _in32 = psim_zip_sync<uint32_t>(in[i][lane]);
            _in32 = psim_shuffle_sync<uint32_t>(_in32, shfl32_ptrn);
            _in[i] = psim_unzip_sync<uint8_t>(_in32, 0);
        }

        uint8_t mix[4];

        mix[0] = psim_shuffle_sync<uint8_t>(_in[0], _in[1], shfl8_lo);
        mix[1] = psim_shuffle_sync<uint8_t>(_in[0], _in[1], shfl8_hi);
        mix[2] = psim_shuffle_sync<uint8_t>(_in[2], _in[3], shfl8_lo);
        mix[3] = psim_shuffle_sync<uint8_t>(_in[2], _in[3], shfl8_hi);

        uint16_t mix16[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            mix16[i] = psim_zip_sync<uint16_t>(mix[i]);
        }

        uint16_t out16[4];
        out16[0] = psim_shuffle_sync<uint16_t>(mix16[0], mix16[2], shfl16_lo);
        out16[1] = psim_shuffle_sync<uint16_t>(mix16[0], mix16[2], shfl16_hi);
        out16[2] = psim_shuffle_sync<uint16_t>(mix16[1], mix16[3], shfl16_lo);
        out16[3] = psim_shuffle_sync<uint16_t>(mix16[1], mix16[3], shfl16_hi);

#pragma unroll
        for (int i = 0; i < 4; i++) {
            out[i][lane] = psim_unzip_sync<uint8_t>(out16[i], 0);
        }
    }
}

__attribute__((noinline)) void interleave4_v3(uint8_t in[4][64],
                                              uint8_t out[4][64]) {
#psim gang_size(GANG_SIZE)
    {
        uint8_t lane = psim_get_lane_num();
        uint8_t rgb[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = i * 64 + lane;
            rgb[i] = in[idx % 4][idx / 4];
        }

#pragma unroll
        for (int i = 0; i < 4; i++) {
            out[i][lane] = rgb[i];
        }
    }
}

int main() {
    assert(GANG_SIZE == 64);
    uint8_t in[4][GANG_SIZE] = {};
    uint8_t out[4][GANG_SIZE] = {};

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < GANG_SIZE; j++) {
            in[i][j] = 64 * i + j;
            out[i][j] = 0;
        }
    }

    int versions = 4;
    const char* version[] = {"simd", "psv_v1", "psv_v2", "psv_v3"};
    bool success = true;
    int reps = 100000;
    for (int v = 0; v < 4; v++) {
        double t0 = GetTimer();
        if (v == 1) {
            for (int rep = 0; rep < reps; rep++) {
                interleave4_v1(in, out);
            }
        } else if (v == 2) {
            for (int rep = 0; rep < reps; rep++) {
                interleave4_v2(in, out);
            }
        } else if (v == 3) {
            for (int rep = 0; rep < reps; rep++) {
                interleave4_v3(in, out);
            }
        } else if (v == 0) {
            for (int rep = 0; rep < reps; rep++) {
                interleave4_simd(in, out);
            }
        }
        double time = GetTimer() - t0;

        int max_err_count = 10;
        int err_count = max_err_count;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < GANG_SIZE; j++) {
                int idx = i * 64 + j;
                uint8_t ref = in[idx % 4][idx / 4];
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
