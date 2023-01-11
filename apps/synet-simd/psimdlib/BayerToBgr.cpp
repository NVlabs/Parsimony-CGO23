#include <assert.h>
#include <parsim.h>
#include "Math.h"
#include "psimdlib.h"

namespace Simd {
namespace Psv {

STATIC_INLINE
void InterleaveBgra(uint8_t in[3], uint8_t alpha, uint8_t out[4]) {
    uint8_t lane = psim_get_lane_num();

    int shfl32_ptrn = (lane * 4) % 16 + (lane / 4);

    int shfl8_lo = lane / 2 + (lane % 2) * 64 + lane / 16 * 8;
    int shfl8_hi = 8 + shfl8_lo;
    int shfl16_lo = lane / 2 + (lane % 2) * 64 + lane / 8 * 4;
    int shfl16_hi = 4 + shfl16_lo;

    uint8_t _in[3];
#pragma unroll
    for (int i = 0; i < 3; i++) {
        uint32_t _in32 = psim_zip_sync<uint32_t>(in[i]);
        _in32 = psim_shuffle_sync<uint32_t>(_in32, shfl32_ptrn);
        _in[i] = psim_unzip_sync<uint8_t>(_in32, 0);
    }

    uint8_t mix[4];

    mix[0] = psim_shuffle_sync<uint8_t>(_in[0], _in[1], shfl8_lo);
    mix[1] = psim_shuffle_sync<uint8_t>(_in[0], _in[1], shfl8_hi);
    mix[2] = psim_shuffle_sync<uint8_t>(_in[2], alpha, shfl8_lo);
    mix[3] = psim_shuffle_sync<uint8_t>(_in[2], alpha, shfl8_hi);

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
        out[i] = psim_unzip_sync<uint8_t>(out16[i], 0);
    }
}

STATIC_INLINE
void InterleaveBgr(uint8_t in[3], uint8_t out[3]) {
    uint8_t lane = psim_get_lane_num();

    uint8_t col8s[3][3];

    int shfl_permute32 = lane % 4 + (lane / 12) * 4;
    int shfl_permute8[3][3] = {};

#pragma unroll
    for (int i = 0; i < 3; i++) {
        uint32_t col32 = psim_zip_sync<uint32_t>(in[i]);
        col32 = psim_shuffle_sync<uint32_t>(col32, shfl_permute32);
#pragma unroll
        for (int j = 0; j < 3; j++) {
            col8s[i][j] = psim_unzip_sync<uint8_t>(col32, j);

            int l = lane + 64 * i;
            shfl_permute8[i][j] =
                (l % 3 == j) ? ((l / 3) % 16) + ((lane / 16) * 16) : -1;
        }
    }

#pragma unroll
    for (int i = 0; i < 3; i++) {
        out[i] = psim_shuffle_sync<uint8_t>(col8s[0][i], shfl_permute8[i][0]) |
                 psim_shuffle_sync<uint8_t>(col8s[1][i], shfl_permute8[i][1]) |
                 psim_shuffle_sync<uint8_t>(col8s[2][i], shfl_permute8[i][2]);
    }
}

STATIC_INLINE uint16_t BayerToGreen(uint16_t greenLeft, uint16_t greenTop,
                                    uint16_t greenRight, uint16_t greenBottom,
                                    uint16_t blueOrRedLeft,
                                    uint16_t blueOrRedTop,
                                    uint16_t blueOrRedRight,
                                    uint16_t blueOrRedBottom) {
    int16_t verticalAbsDifference =
        AbsDiff((int16_t)blueOrRedTop, (int16_t)blueOrRedBottom);
    int16_t horizontalAbsDifference =
        AbsDiff((int16_t)blueOrRedLeft, (int16_t)blueOrRedRight);

    uint16_t green = Average(greenLeft, greenTop, greenRight, greenBottom);
    green = horizontalAbsDifference > verticalAbsDifference
                ? Average(greenTop, greenBottom)
                : green;

    return verticalAbsDifference > horizontalAbsDifference
               ? Average(greenRight, greenLeft)
               : green;
}

template <int index, int part>
STATIC_INLINE int16_t Get(const uint16_t src[2][12]) {
    return src[part][index];
}

STATIC_INLINE uint8_t Merge16(uint16_t a, uint16_t b) {
    uint16_t val = (b << 8) | a;
    return psim_unzip_sync<uint8_t>(val, 0);
}

template <SimdPixelFormatType bayerFormat>
STATIC_INLINE void BayerToBgr(const uint16_t s[2][12], uint8_t d[6]);

template <>
STATIC_INLINE void BayerToBgr<SimdPixelFormatBayerGrbg>(const uint16_t s[2][12],
                                                        uint8_t d[6]) {
    d[0] = Merge16(
        Average(Get<0, 1>(s), Get<7, 0>(s)),
        Average(Get<0, 1>(s), Get<2, 1>(s), Get<7, 0>(s), Get<8, 0>(s)));

    d[1] = Merge16(
        Get<4, 0>(s),
        BayerToGreen(Get<4, 0>(s), Get<2, 0>(s), Get<5, 0>(s), Get<7, 1>(s),
                     Get<3, 1>(s), Get<1, 1>(s), Get<5, 1>(s), Get<11, 0>(s)));

    d[2] = Merge16(Average(Get<3, 1>(s), Get<4, 1>(s)), Get<4, 1>(s));
    d[3] = Merge16(Get<7, 0>(s), Average(Get<7, 0>(s), Get<8, 0>(s)));
    d[4] = Merge16(
        BayerToGreen(Get<6, 1>(s), Get<4, 0>(s), Get<7, 1>(s), Get<9, 1>(s),
                     Get<6, 0>(s), Get<0, 1>(s), Get<8, 0>(s), Get<10, 0>(s)),
        Get<7, 1>(s));
    d[5] = Merge16(
        Average(Get<3, 1>(s), Get<4, 1>(s), Get<9, 0>(s), Get<11, 0>(s)),
        Average(Get<4, 1>(s), Get<11, 0>(s)));
}

template <>
SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerGbrg>(const uint16_t s[2][12],
                                                      uint8_t d[6]) {
    d[0] = Merge16(Average(Get<3, 1>(s), Get<4, 1>(s)), Get<4, 1>(s));
    d[1] = Merge16(
        Get<4, 0>(s),
        BayerToGreen(Get<4, 0>(s), Get<2, 0>(s), Get<5, 0>(s), Get<7, 1>(s),
                     Get<3, 1>(s), Get<1, 1>(s), Get<5, 1>(s), Get<11, 0>(s)));
    d[2] = Merge16(
        Average(Get<0, 1>(s), Get<7, 0>(s)),
        Average(Get<0, 1>(s), Get<2, 1>(s), Get<7, 0>(s), Get<8, 0>(s)));
    d[3] = Merge16(
        Average(Get<3, 1>(s), Get<4, 1>(s), Get<9, 0>(s), Get<11, 0>(s)),
        Average(Get<4, 1>(s), Get<11, 0>(s)));
    d[4] = Merge16(
        BayerToGreen(Get<6, 1>(s), Get<4, 0>(s), Get<7, 1>(s), Get<9, 1>(s),
                     Get<6, 0>(s), Get<0, 1>(s), Get<8, 0>(s), Get<10, 0>(s)),
        Get<7, 1>(s));
    d[5] = Merge16(Get<7, 0>(s), Average(Get<7, 0>(s), Get<8, 0>(s)));
}

template <>
SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerRggb>(const uint16_t s[2][12],
                                                      uint8_t d[6]) {
    d[0] =
        Merge16(Average(Get<0, 0>(s), Get<2, 0>(s), Get<6, 1>(s), Get<7, 1>(s)),
                Average(Get<2, 0>(s), Get<7, 1>(s)));
    d[1] = Merge16(
        BayerToGreen(Get<3, 1>(s), Get<0, 1>(s), Get<4, 1>(s), Get<7, 0>(s),
                     Get<3, 0>(s), Get<1, 0>(s), Get<5, 0>(s), Get<9, 1>(s)),
        Get<4, 1>(s));
    d[2] = Merge16(Get<4, 0>(s), Average(Get<4, 0>(s), Get<5, 0>(s)));
    d[3] = Merge16(Average(Get<6, 1>(s), Get<7, 1>(s)), Get<7, 1>(s));
    d[4] = Merge16(
        Get<7, 0>(s),
        BayerToGreen(Get<7, 0>(s), Get<4, 1>(s), Get<8, 0>(s), Get<11, 0>(s),
                     Get<6, 1>(s), Get<2, 0>(s), Get<8, 1>(s), Get<10, 1>(s)));
    d[5] = Merge16(
        Average(Get<4, 0>(s), Get<9, 1>(s)),
        Average(Get<4, 0>(s), Get<5, 0>(s), Get<9, 1>(s), Get<11, 1>(s)));
}

template <>
SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerBggr>(const uint16_t s[2][12],
                                                      uint8_t d[6]) {
    d[0] = Merge16(Get<4, 0>(s), Average(Get<4, 0>(s), Get<5, 0>(s)));
    d[1] = Merge16(
        BayerToGreen(Get<3, 1>(s), Get<0, 1>(s), Get<4, 1>(s), Get<7, 0>(s),
                     Get<3, 0>(s), Get<1, 0>(s), Get<5, 0>(s), Get<9, 1>(s)),
        Get<4, 1>(s));
    d[2] =
        Merge16(Average(Get<0, 0>(s), Get<2, 0>(s), Get<6, 1>(s), Get<7, 1>(s)),
                Average(Get<2, 0>(s), Get<7, 1>(s)));
    d[3] = Merge16(
        Average(Get<4, 0>(s), Get<9, 1>(s)),
        Average(Get<4, 0>(s), Get<5, 0>(s), Get<9, 1>(s), Get<11, 1>(s)));
    d[4] = Merge16(
        Get<7, 0>(s),
        BayerToGreen(Get<7, 0>(s), Get<4, 1>(s), Get<8, 0>(s), Get<11, 0>(s),
                     Get<6, 1>(s), Get<2, 0>(s), Get<8, 1>(s), Get<10, 1>(s)));
    d[5] = Merge16(Average(Get<6, 1>(s), Get<7, 1>(s)), Get<7, 1>(s));
}

#define GS 64

template <SimdPixelFormatType bayerFormat, bool isAlpha>
void BayerToBgr(const uint8_t* bayer, size_t width, size_t height,
                size_t bayerStride, uint8_t* bgr, size_t bgrStride,
                uint8_t alpha = 0) {
    const uint8_t* src[3];

    for (size_t row = 0; row < height; row += 2) {
        src[0] = (row == 0 ? bayer : bayer - 2 * bayerStride);
        src[1] = bayer;
        src[2] = (row == height - 2 ? bayer : bayer + 2 * bayerStride);

        size_t num_gangs = CeilDiv(width, (size_t)GS);
#psim num_spmd_gangs(num_gangs) gang_size(GS)
        {
            size_t col = psim_get_thread_num();
            size_t lane = psim_get_lane_num();
            uint16_t src16[2][12] = {};
            uint8_t dst[6] = {};

            const int SRC_ID[12] = {0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2};
            const int STRIDE_FACTOR[12] = {1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0};
            const int DELTA[12] = {-1, 0, 1, -2, 0, 2, -2, 0, +2, -1, 0, 1};

            size_t base_col_gang =
                psim_get_gang_num() * psim_get_gang_size();
#pragma unroll
            for (int i = 0; i < 12; i++) {
                int fold = 0;
                int64_t sum = (int64_t)lane + DELTA[i];
                if (psim_is_head_gang()) {
                    if (sum < 0) {
                        fold = 2;
                    }
                } else if (psim_is_tail_gang()) {
                    base_col_gang = width - psim_get_gang_size();
                    col = base_col_gang + lane;
                    if (sum >= psim_get_gang_size()) {
                        fold = -2;
                    }
                }

                size_t index =
                    col + STRIDE_FACTOR[i] * bayerStride + DELTA[i] + fold;
                uint8_t val = *(src[SRC_ID[i]] + index);

                uint16_t val16 = psim_zip_sync<uint16_t>(val);
                src16[0][i] = val16 & 0x00ff;
                src16[1][i] = (val16 >> 8) & 0x00ff;
            }

            BayerToBgr<bayerFormat>(src16, dst);

#pragma unroll
            for (int i = 0; i < 2; i++) {
                uint8_t out8[4];
                int out_elems;
                if (isAlpha) {
                    out_elems = 4;
                    InterleaveBgra(dst + i * 3, alpha, out8);
                } else {
                    out_elems = 3;
                    InterleaveBgr(dst + i * 3, out8);
                }

#pragma unroll
                for (int j = 0; j < out_elems; j++) {
                    size_t out_index = out_elems * base_col_gang +
                                       bgrStride * i +
                                       j * psim_get_gang_size() + lane;

                    bgr[out_index] = out8[j];
                }
            }
        }
        bayer += 2 * bayerStride;
        bgr += 2 * bgrStride;
    }
}

void BayerToBgr(const uint8_t* bayer, size_t width, size_t height,
                size_t bayerStride, SimdPixelFormatType bayerFormat,
                uint8_t* bgr, size_t bgrStride) {
    switch (bayerFormat) {
        case SimdPixelFormatBayerGrbg:
            BayerToBgr<SimdPixelFormatBayerGrbg, false>(
                bayer, width, height, bayerStride, bgr, bgrStride);
            break;
        case SimdPixelFormatBayerGbrg:
            BayerToBgr<SimdPixelFormatBayerGbrg, false>(
                bayer, width, height, bayerStride, bgr, bgrStride);
            break;
        case SimdPixelFormatBayerRggb:
            BayerToBgr<SimdPixelFormatBayerRggb, false>(
                bayer, width, height, bayerStride, bgr, bgrStride);
            break;
        case SimdPixelFormatBayerBggr:
            BayerToBgr<SimdPixelFormatBayerBggr, false>(
                bayer, width, height, bayerStride, bgr, bgrStride);
            break;
        default:
            assert(0);
    }
}

void BayerToBgra(const uint8_t* bayer, size_t width, size_t height,
                 size_t bayerStride, SimdPixelFormatType bayerFormat,
                 uint8_t* bgra, size_t bgraStride, uint8_t alpha) {
    switch (bayerFormat) {
        case SimdPixelFormatBayerGrbg:
            BayerToBgr<SimdPixelFormatBayerGrbg, true>(
                bayer, width, height, bayerStride, bgra, bgraStride, alpha);
            break;
        case SimdPixelFormatBayerGbrg:
            BayerToBgr<SimdPixelFormatBayerGbrg, true>(
                bayer, width, height, bayerStride, bgra, bgraStride, alpha);
            break;
        case SimdPixelFormatBayerRggb:
            BayerToBgr<SimdPixelFormatBayerRggb, true>(
                bayer, width, height, bayerStride, bgra, bgraStride, alpha);
            break;
        case SimdPixelFormatBayerBggr:
            BayerToBgr<SimdPixelFormatBayerBggr, true>(
                bayer, width, height, bayerStride, bgra, bgraStride, alpha);
            break;
        default:
            assert(0);
    }
}

}  // namespace Psv
}  // namespace Simd
