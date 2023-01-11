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
#include <sys/time.h>
#include <time.h>

#include <cassert>

#include "Simd/SimdCompare.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdStore.h"

#define STATIC_INLINE inline __attribute__((always_inline))
/* return time in microseconds */
static __attribute__((unused)) double GetTimer() {
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return ((double)1000000 * (double)tv.tv_sec + (double)tv.tv_usec);
}

#define WS 64

template <class T>
STATIC_INLINE char GetChar(T value, size_t index) {
    return ((char*)&value)[index];
}

const size_t A = sizeof(__m512i);
const size_t F = sizeof(__m512) / sizeof(float);

STATIC_INLINE __mmask64 TailMask64(ptrdiff_t tail) {
    return tail <= 0
               ? __mmask64(0)
               : (tail >= 64 ? __mmask64(-1) : __mmask64(-1) >> (64 - tail));
}

STATIC_INLINE __mmask16 TailMask16(ptrdiff_t tail) {
    return tail <= 0
               ? __mmask16(0)
               : (tail >= 16 ? __mmask16(-1) : __mmask16(-1) >> (16 - tail));
}

namespace Simd {
namespace Avx512bw {
template <bool mask>
STATIC_INLINE void Unpack_avx(const uint8_t* sum, const __m512i& area,
                              uint32_t* s0a0, const __mmask16* tailMasks) {
    const __m512i _sum =
        _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<true>(sum)));
    const __m512i saLo = _mm512_unpacklo_epi8(_sum, area);
    const __m512i saHi = _mm512_unpackhi_epi8(_sum, area);
    Store<true, mask>(s0a0 + 0 * F, _mm512_unpacklo_epi8(saLo, K_ZERO),
                      tailMasks[0]);
    Store<true, mask>(s0a0 + 1 * F, _mm512_unpackhi_epi8(saLo, K_ZERO),
                      tailMasks[1]);
    Store<true, mask>(s0a0 + 2 * F, _mm512_unpacklo_epi8(saHi, K_ZERO),
                      tailMasks[2]);
    Store<true, mask>(s0a0 + 3 * F, _mm512_unpackhi_epi8(saHi, K_ZERO),
                      tailMasks[3]);
}

}  // namespace Avx512bw
}  // namespace Simd

STATIC_INLINE void Unpack_simple(const uint8_t* sa, uint8_t area, size_t width,
                                 uint32_t* dst_psv_simple) {
#psim num_spmd_threads(width) gang_size(WS)
    {
        size_t col = psim_get_thread_num();
        dst_psv_simple[col] = ((uint32_t)area) << 16 | (uint32_t)sa[col];
    }
}

STATIC_INLINE void Unpack_complex(const uint8_t* sa, uint8_t area,
                                  size_t num_gangs, size_t width,
                                  uint8_t* sa32_ptr8, uint32_t* sa32) {
#psim num_spmd_gangs(num_gangs) gang_size(WS)
    {
        uint32_t lane = psim_get_lane_num();
        size_t col = psim_get_thread_num();
        size_t gang_num = psim_get_gang_num();
        uint8_t my_sa = 0;
        uint8_t my_zero = 0;
        uint8_t my_area = area;

        int shfl32 = (lane * 4) % 16 + (lane / 4);
        int shfl8_lo = lane / 2 + (lane % 2) * 64 + lane / 16 * 8;
        int shfl8_hi = 8 + shfl8_lo;
        if (psim_is_tail_gang()) {
            if (col < width) my_sa = sa[col];
        } else {
            my_sa = sa[col];
        }
        uint32_t val32 = psim_zip_sync<uint32_t>(my_sa);
        uint32_t permute32 = psim_shuffle_sync<uint32_t>(val32, shfl32);
        uint8_t permute8 = psim_unzip_sync<uint8_t>(permute32, 0);

        uint8_t saLo = psim_shuffle_sync<uint8_t>(permute8, my_area, shfl8_lo);
        uint8_t saHi = psim_shuffle_sync<uint8_t>(permute8, my_area, shfl8_hi);

        uint8_t my_sa8_0 = psim_shuffle_sync<uint8_t>(saLo, my_zero, shfl8_lo);
        uint8_t my_sa8_1 = psim_shuffle_sync<uint8_t>(saLo, my_zero, shfl8_hi);
        uint8_t my_sa8_2 = psim_shuffle_sync<uint8_t>(saHi, my_zero, shfl8_lo);
        uint8_t my_sa8_3 = psim_shuffle_sync<uint8_t>(saHi, my_zero, shfl8_hi);

        if (psim_is_tail_gang()) {
            if (((256 * gang_num) + lane < width * 4))
                sa32_ptr8[(256 * gang_num) + lane] = my_sa8_0;
            if (((256 * gang_num) + lane + 64 < width * 4))
                sa32_ptr8[(256 * gang_num) + lane + 64] = my_sa8_1;
            if (((256 * gang_num) + lane + 128 < width * 4))
                sa32_ptr8[(256 * gang_num) + lane + 128] = my_sa8_2;
            if (((256 * gang_num) + lane + 192 < width * 4))
                sa32_ptr8[(256 * gang_num) + lane + 192] = my_sa8_3;
        } else {
            sa32_ptr8[(256 * gang_num) + lane] = my_sa8_0;
            sa32_ptr8[(256 * gang_num) + lane + 64] = my_sa8_1;
            sa32_ptr8[(256 * gang_num) + lane + 128] = my_sa8_2;
            sa32_ptr8[(256 * gang_num) + lane + 192] = my_sa8_3;
        }
    }
}

int main() {
    size_t array_size = 1929;
    uint8_t* a = (uint8_t*)malloc(array_size * sizeof(uint8_t));
    uint8_t b = rand() % 0xFF;
    uint32_t* c = (uint32_t*)malloc(array_size * sizeof(uint32_t));
    uint32_t* dst_avx = (uint32_t*)malloc(array_size * sizeof(uint32_t));
    uint32_t* dst_psv_simple = (uint32_t*)malloc(array_size * sizeof(uint32_t));
    uint32_t* dst_psv_complex =
        (uint32_t*)malloc(array_size * sizeof(uint32_t));
    uint8_t* dst_psv_complex_ptr8 = (uint8_t*)dst_psv_complex;

    for (int i = 0; i < array_size; i++) {
        a[i] = i % 0xFF;
        c[i] = (uint32_t)b << 16 | (uint32_t)a[i];
    }

    size_t alignedWidth = array_size / A * A;
    __mmask64 tailMask = TailMask64(array_size - alignedWidth);
    __mmask16 tailMasks[4];
    for (size_t c = 0; c < 4; ++c)
        tailMasks[c] = TailMask16(array_size - alignedWidth - F * c);

    size_t col = 0;
    double time_sum = 0;
    for (int row = 0; row < 1080; row++) {
        double t0 = GetTimer();

        __m512i _area = _mm512_set1_epi8(b);
        for (col = 0; col < alignedWidth; col += A)
            Simd::Avx512bw::Unpack_avx<false>(a + col, _area, dst_avx + col,
                                              tailMasks);
        if (col < array_size)
            Simd::Avx512bw::Unpack_avx<true>(a + col, _area, dst_avx + col,
                                             tailMasks);

        time_sum += GetTimer() - t0;
    }
    printf("time_avx: %.2f\n", time_sum);

    time_sum = 0;
    size_t num_gangs = ((array_size + (WS - 1)) / WS);
    for (int row = 0; row < 1080; row++) {
        double t0 = GetTimer();
        Unpack_simple(a, b, array_size, dst_psv_simple);
        time_sum += GetTimer() - t0;
    }
    printf("time_psv_simple: %.2f\n", time_sum);

    time_sum = 0;
    for (int row = 0; row < 1080; row++) {
        double t0 = GetTimer();
        Unpack_complex(a, b, num_gangs, array_size, dst_psv_complex_ptr8,
                       dst_psv_simple);
        time_sum += GetTimer() - t0;
    }
    printf("time_psv_complex: %.2f\n", time_sum);

    bool error = false;
    for (int i = 0; i < array_size; i++) {
        if (c[i] != dst_avx[i]) {
            printf("Error_avx @%d - %d != %d \n", i, c[i], dst_avx[i]);
            error = true;
        }
    }

    for (int i = 0; i < array_size; i++) {
        if (c[i] != dst_psv_simple[i]) {
            printf("Error_psv_simple @%d - %d != %d \n", i, c[i],
                   dst_psv_simple[i]);
            error = true;
        }
    }

    for (int i = 0; i < array_size; i++) {
        if (c[i] != dst_psv_complex[i]) {
            printf("Error_psv_complex @%d - %d != %d \n", i, c[i],
                   dst_psv_complex[i]);
            error = true;
        }
    }

    assert(!error);
    printf("Success!\n");
}
