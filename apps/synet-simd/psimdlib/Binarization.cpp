#include <assert.h>
#include <parsim.h>
#include "Math.h"
#include "psimdlib.h"

namespace Simd {
namespace Psv {

#define GS 64

template <SimdCompareType compareType>
void Binarization(const uint8_t* src, size_t srcStride, size_t width,
                  size_t height, uint8_t value, uint8_t positive,
                  uint8_t negative, uint8_t* dst, size_t dstStride) {
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(GS)
        {
            size_t col = psim_get_thread_num();
            dst[col] =
                Compare8u<compareType>(src[col], value) ? positive : negative;
        }
        src += srcStride;
        dst += dstStride;
    }
}

void Binarization(const uint8_t* src, size_t srcStride, size_t width,
                  size_t height, uint8_t value, uint8_t positive,
                  uint8_t negative, uint8_t* dst, size_t dstStride,
                  SimdCompareType compareType) {
    switch (compareType) {
        case SimdCompareEqual:
            return Binarization<SimdCompareEqual>(src, srcStride, width, height,
                                                  value, positive, negative,
                                                  dst, dstStride);
        case SimdCompareNotEqual:
            return Binarization<SimdCompareNotEqual>(src, srcStride, width,
                                                     height, value, positive,
                                                     negative, dst, dstStride);
        case SimdCompareGreater:
            return Binarization<SimdCompareGreater>(src, srcStride, width,
                                                    height, value, positive,
                                                    negative, dst, dstStride);
        case SimdCompareGreaterOrEqual:
            return Binarization<SimdCompareGreaterOrEqual>(
                src, srcStride, width, height, value, positive, negative, dst,
                dstStride);
        case SimdCompareLesser:
            return Binarization<SimdCompareLesser>(src, srcStride, width,
                                                   height, value, positive,
                                                   negative, dst, dstStride);
        case SimdCompareLesserOrEqual:
            return Binarization<SimdCompareLesserOrEqual>(
                src, srcStride, width, height, value, positive, negative, dst,
                dstStride);
        default:
            assert(0);
    }
}

STATIC_INLINE void Unpack(const uint8_t * sa, uint8_t area, size_t num_gangs, size_t width, uint8_t * sa32_ptr8, uint32_t* sa32){
/* UNPACK */      

#psim num_spmd_threads(width) gang_size(GS)
    {
        size_t col = psim_get_thread_num();
        sa32[col] = ((uint32_t)area) << 16 | (uint32_t)sa[col];
    }

/* Alternate approach with same performance:
#psim num_spmd_gangs(num_gangs) gang_size(GS)
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
            if(col < width)
                my_sa = sa[col];
        }
        else{
            my_sa = sa[col];
        }
        uint32_t val32 = psim_zip_sync<uint32_t>(my_sa);
        uint32_t permute32 = psim_shuffle_sync<uint32_t>(val32, shfl32);
        uint8_t permute8 = psim_unzip_sync<uint8_t>(permute32, 0);

        uint8_t saLo =
            psim_shuffle_sync<uint8_t>(permute8, my_area, shfl8_lo);
        uint8_t saHi =
            psim_shuffle_sync<uint8_t>(permute8, my_area, shfl8_hi);

        uint8_t my_sa8_0 =
            psim_shuffle_sync<uint8_t>(saLo, my_zero, shfl8_lo);
        uint8_t my_sa8_1 =
            psim_shuffle_sync<uint8_t>(saLo, my_zero, shfl8_hi);
        uint8_t my_sa8_2 =
            psim_shuffle_sync<uint8_t>(saHi, my_zero, shfl8_lo);
        uint8_t my_sa8_3 =
            psim_shuffle_sync<uint8_t>(saHi, my_zero, shfl8_hi);
        
        if(psim_is_tail_gang()){
            if(((256*gang_num) + lane < width*4))
                sa32_ptr8[(256*gang_num) + lane] = my_sa8_0;
            if(((256*gang_num) + lane + 64 < width*4))
                sa32_ptr8[(256*gang_num) + lane + 64] = my_sa8_1;
            if(((256*gang_num) + lane + 128 < width*4))
                sa32_ptr8[(256*gang_num) + lane + 128] = my_sa8_2;
            if(((256*gang_num) + lane + 192 < width*4))
                sa32_ptr8[(256*gang_num) + lane + 192] = my_sa8_3;
        }
        else{
            sa32_ptr8[(256*gang_num) + lane] = my_sa8_0;
            sa32_ptr8[(256*gang_num) + lane + 64] = my_sa8_1;
            sa32_ptr8[(256*gang_num) + lane + 128] = my_sa8_2;
            sa32_ptr8[(256*gang_num) + lane + 192] = my_sa8_3;
        }
    }*/
}

SIMD_INLINE void getResult(uint32_t * sum, uint8_t positive, uint8_t negative, uint8_t threshold, size_t width, uint8_t * dst){
/* getResult */       
#psim num_spmd_threads(width) gang_size(GS)
    {
        size_t col = psim_get_thread_num();

        dst[col] = ((uint16_t)sum[col] * 0xFF > threshold*(uint16_t)(sum[col]>>16)) ? positive : negative;

/*      uint32_t my_dst = psim_madd16x2((uint16_t)sum[col] * 0xFF , (uint16_t)0xFF,  
                                    (uint16_t) - threshold, (uint16_t)(sum[col]>>16));
        dst[col] = my_dst > 0 ? positive : negative;*/
    }
}


template <SimdCompareType compareType>
STATIC_INLINE uint8_t GetSa(uint8_t src, uint8_t value) {
    return Compare8u<compareType>(src, value) ? 0x01 : 0x00;
}

template <SimdCompareType compareType>
void AveragingBinarization(const uint8_t* src, size_t srcStride, size_t width,
                           size_t height, uint8_t value, size_t neighborhood,
                           uint8_t threshold, uint8_t positive,
                           uint8_t negative, uint8_t* dst, size_t dstStride) {
    assert(width > neighborhood && height > neighborhood &&
           neighborhood < 0x80);

    size_t edge = neighborhood + 1;
    uint8_t* sa = (uint8_t*)calloc(width , sizeof(uint8_t));


    uint32_t* sa32 = (uint32_t*)calloc(width + 2 * edge, sizeof(uint32_t));
    sa32 = (uint32_t*)sa32 + edge;
    uint8_t* sa32_ptr8 = (uint8_t*)sa32;


    uint32_t* sum32 = (uint32_t*)calloc(width + 2 * edge, sizeof(uint32_t));
    sum32 = (uint32_t*)sum32 + edge;


    uint8_t area = 0;
    size_t num_gangs = CeilDiv(width, (size_t)GS);
    for (size_t row = 0; row < neighborhood; ++row) {
        area++;
        const uint8_t* s = src + row * srcStride;
#psim num_spmd_threads(width) gang_size(GS)
        {
            size_t col = psim_get_thread_num();
            sa[col] += GetSa<compareType>(s[col], value);
        }
    }

    for (size_t row = 0; row < height; ++row) {
        if (row < height - neighborhood) {
            area++;
            const uint8_t* s = src + (row + neighborhood) * srcStride;
#psim num_spmd_threads(width) gang_size(GS)
            {
                size_t col = psim_get_thread_num();
                sa[col] += GetSa<compareType>(s[col], value);
            }
        }

        if (row > neighborhood) {
            area--;
            const uint8_t* s = src + (row - neighborhood - 1) * srcStride;
#psim num_spmd_threads(width) gang_size(GS)
            {
                size_t col = psim_get_thread_num();
                sa[col] -= GetSa<compareType>(s[col], value);
            }
        }
        Unpack(sa, area, num_gangs, width, sa32_ptr8, sa32);

        uint32_t saSum32 = 0;
        for (size_t col = 0; col < neighborhood; ++col){
            saSum32 += sa32[col];
        }
        for (size_t col = 0; col < width; ++col) {
            saSum32 += sa32[col + neighborhood];
            saSum32 -= sa32[col - neighborhood - 1];
            sum32[col] = saSum32;
        }

        getResult(sum32, positive, negative, threshold, width, dst);
        dst += dstStride;
    }
}

void AveragingBinarization(const uint8_t* src, size_t srcStride, size_t width,
                           size_t height, uint8_t value, size_t neighborhood,
                           uint8_t threshold, uint8_t positive,
                           uint8_t negative, uint8_t* dst, size_t dstStride,
                           SimdCompareType compareType) {
    switch (compareType) {
        case SimdCompareEqual:
            return AveragingBinarization<SimdCompareEqual>(
                src, srcStride, width, height, value, neighborhood, threshold,
                positive, negative, dst, dstStride);
        case SimdCompareNotEqual:
            return AveragingBinarization<SimdCompareNotEqual>(
                src, srcStride, width, height, value, neighborhood, threshold,
                positive, negative, dst, dstStride);
        case SimdCompareGreater:
            return AveragingBinarization<SimdCompareGreater>(
                src, srcStride, width, height, value, neighborhood, threshold,
                positive, negative, dst, dstStride);
        case SimdCompareGreaterOrEqual:
            return AveragingBinarization<SimdCompareGreaterOrEqual>(
                src, srcStride, width, height, value, neighborhood, threshold,
                positive, negative, dst, dstStride);
        case SimdCompareLesser:
            return AveragingBinarization<SimdCompareLesser>(
                src, srcStride, width, height, value, neighborhood, threshold,
                positive, negative, dst, dstStride);
        case SimdCompareLesserOrEqual:
            return AveragingBinarization<SimdCompareLesserOrEqual>(
                src, srcStride, width, height, value, neighborhood, threshold,
                positive, negative, dst, dstStride);
        default:
            assert(0);
    }
}

}  // namespace Psv
}  // namespace Simd
