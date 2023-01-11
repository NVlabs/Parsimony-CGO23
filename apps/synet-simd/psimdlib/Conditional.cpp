#include <assert.h>
#include <parsim.h>
#include "Math.h"
#include "psimdlib.h"


namespace Simd{
namespace Psv{

#define GS 64

template <SimdCompareType compareType>
void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, uint32_t * count)
{
    *count = 0;
    PsimCollectiveAddAbsDiff<uint32_t> _sum;
    for (size_t row = 0; row < height; ++row)
    {
#psim num_spmd_threads(width) gang_size(GS)
        {
            size_t col = psim_get_thread_num();
            if (Compare8u<compareType>(src[col], value))
                _sum.AddAbsDiff((uint8_t)1, (uint8_t)0);
        }
        src += stride;
    }
    *count = _sum.ReduceSum();
}

void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height,
    uint8_t value, SimdCompareType compareType, uint32_t * count)
{
    switch (compareType)
    {
    case SimdCompareEqual:
        return ConditionalCount8u<SimdCompareEqual>(src, stride, width, height, value, count);
    case SimdCompareNotEqual:
        return ConditionalCount8u<SimdCompareNotEqual>(src, stride, width, height, value, count);
    case SimdCompareGreater:
        return ConditionalCount8u<SimdCompareGreater>(src, stride, width, height, value, count);
    case SimdCompareGreaterOrEqual:
        return ConditionalCount8u<SimdCompareGreaterOrEqual>(src, stride, width, height, value, count);
    case SimdCompareLesser:
        return ConditionalCount8u<SimdCompareLesser>(src, stride, width, height, value, count);
    case SimdCompareLesserOrEqual:
        return ConditionalCount8u<SimdCompareLesserOrEqual>(src, stride, width, height, value, count);
    default:
        assert(0);
    }
}

template <SimdCompareType compareType>
void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height, int16_t value, uint32_t * count)
{
    *count = 0;
    PsimCollectiveAddAbsDiff<uint32_t> _sum;
    for (size_t row = 0; row < height; ++row)
    {
        const int16_t * s = (const int16_t *)src;
#psim num_spmd_threads(width) gang_size(GS)
        {
            size_t col = psim_get_thread_num();
            if (Compare16i<compareType>(s[col], value))
                _sum.AddAbsDiff((uint8_t)1, (uint8_t)0);
        }
        src += stride;
    }
    *count = _sum.ReduceSum();
}

void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height,
    int16_t value, SimdCompareType compareType, uint32_t * count)
{
    switch (compareType)
    {
    case SimdCompareEqual:
        return ConditionalCount16i<SimdCompareEqual>(src, stride, width, height, value, count);
    case SimdCompareNotEqual:
        return ConditionalCount16i<SimdCompareNotEqual>(src, stride, width, height, value, count);
    case SimdCompareGreater:
        return ConditionalCount16i<SimdCompareGreater>(src, stride, width, height, value, count);
    case SimdCompareGreaterOrEqual:
        return ConditionalCount16i<SimdCompareGreaterOrEqual>(src, stride, width, height, value, count);
    case SimdCompareLesser:
        return ConditionalCount16i<SimdCompareLesser>(src, stride, width, height, value, count);
    case SimdCompareLesserOrEqual:
        return ConditionalCount16i<SimdCompareLesserOrEqual>(src, stride, width, height, value, count);
    default:
        assert(0);
    }
}

template <SimdCompareType compareType>
void ConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
    const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
{
    PsimCollectiveAddAbsDiff<uint64_t> _sum;
    for (size_t row = 0; row < height; ++row)
    {
#psim num_spmd_threads(width) gang_size(GS)
        {
            size_t col = psim_get_thread_num();
            if (Compare8u<compareType>(mask[col], value)){
                _sum.AddAbsDiff(src[col], (uint8_t)0);
            }
        }
        src += srcStride;
        mask += maskStride;
    }
    *sum = _sum.ReduceSum();
}

void ConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
    const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
{
    switch (compareType)
    {
    case SimdCompareEqual:
        return ConditionalSum<SimdCompareEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
    case SimdCompareNotEqual:
        return ConditionalSum<SimdCompareNotEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
    case SimdCompareGreater:
        return ConditionalSum<SimdCompareGreater>(src, srcStride, width, height, mask, maskStride, value, sum);
    case SimdCompareGreaterOrEqual:
        return ConditionalSum<SimdCompareGreaterOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
    case SimdCompareLesser:
        return ConditionalSum<SimdCompareLesser>(src, srcStride, width, height, mask, maskStride, value, sum);
    case SimdCompareLesserOrEqual:
        return ConditionalSum<SimdCompareLesserOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
    default:
        assert(0);
    }
}

template <SimdCompareType compareType>
void ConditionalFill(const uint8_t * src, size_t srcStride, size_t width, size_t height,
    uint8_t threshold, uint8_t value, uint8_t * dst, size_t dstStride)
{
    for (size_t row = 0; row < height; ++row)
    {
#psim num_spmd_threads(width) gang_size(GS)
        {
            size_t col = psim_get_thread_num();
            if (Compare8u<compareType>(src[col], threshold))
                dst[col] = value;
        }
        src += srcStride;
        dst += dstStride;
    }
}

void ConditionalFill(const uint8_t * src, size_t srcStride, size_t width, size_t height,
    uint8_t threshold, SimdCompareType compareType, uint8_t value, uint8_t * dst, size_t dstStride)
{
    switch (compareType)
    {
    case SimdCompareEqual:
        return ConditionalFill<SimdCompareEqual>(src, srcStride, width, height, threshold, value, dst, dstStride);
    case SimdCompareNotEqual:
        return ConditionalFill<SimdCompareNotEqual>(src, srcStride, width, height, threshold, value, dst, dstStride);
    case SimdCompareGreater:
        return ConditionalFill<SimdCompareGreater>(src, srcStride, width, height, threshold, value, dst, dstStride);
    case SimdCompareGreaterOrEqual:
        return ConditionalFill<SimdCompareGreaterOrEqual>(src, srcStride, width, height, threshold, value, dst, dstStride);
    case SimdCompareLesser:
        return ConditionalFill<SimdCompareLesser>(src, srcStride, width, height, threshold, value, dst, dstStride);
    case SimdCompareLesserOrEqual:
        return ConditionalFill<SimdCompareLesserOrEqual>(src, srcStride, width, height, threshold, value, dst, dstStride);
    default:
        assert(0);
    }
}
    
}  // namespace Psv
}  // namespace Simd