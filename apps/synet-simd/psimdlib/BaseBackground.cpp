#include <assert.h>
#include <parsim.h>
#include "Math.h"
#include "psimdlib.h"

namespace Simd {
namespace Psv {

void BackgroundGrowRangeSlow(const uint8_t* value, size_t valueStride,
                             size_t width, size_t height, uint8_t* lo,
                             size_t loStride, uint8_t* hi, size_t hiStride) {
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(64)
        {
            size_t col = psim_get_thread_num();
            if (value[col] < lo[col]) lo[col]--;
            if (value[col] > hi[col]) hi[col]++;
        }
        value += valueStride;
        lo += loStride;
        hi += hiStride;
    }
}

void BackgroundGrowRangeFast(const uint8_t* value, size_t valueStride,
                             size_t width, size_t height, uint8_t* lo,
                             size_t loStride, uint8_t* hi, size_t hiStride) {
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(64)
        {
            size_t col = psim_get_thread_num();
            lo[col] = std::min(lo[col], value[col]);
            hi[col] = std::max(hi[col], value[col]);
        }
        value += valueStride;
        lo += loStride;
        hi += hiStride;
    }
}

void BackgroundIncrementCount(const uint8_t* value, size_t valueStride,
                              size_t width, size_t height,
                              const uint8_t* loValue, size_t loValueStride,
                              const uint8_t* hiValue, size_t hiValueStride,
                              uint8_t* loCount, size_t loCountStride,
                              uint8_t* hiCount, size_t hiCountStride) {
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(64)
        {
            size_t col = psim_get_thread_num();
            if (value[col] < loValue[col])
                loCount[col] = psim_uadd_sat(loCount[col], (uint8_t)1);
            if (value[col] > hiValue[col])
                hiCount[col] = psim_uadd_sat(hiCount[col], (uint8_t)1);
        }
        value += valueStride;
        loValue += loValueStride;
        hiValue += hiValueStride;
        loCount += loCountStride;
        hiCount += hiCountStride;
    }
}

STATIC_INLINE void AdjustLo(const uint8_t& count, uint8_t& value,
                            int threshold) {
    if (count > threshold) {
        value = psim_usub_sat(value, (uint8_t)1);
    } else if (count < threshold) {
        value = psim_uadd_sat(value, (uint8_t)1);
    }
}

STATIC_INLINE void AdjustHi(const uint8_t& count, uint8_t& value,
                            int threshold) {
    if (count > threshold) {
        value = psim_uadd_sat(value, (uint8_t)1);
    } else if (count < threshold) {
        value = psim_usub_sat(value, (uint8_t)1);
    }
}

void BackgroundAdjustRange(uint8_t* loCount, size_t loCountStride, size_t width,
                           size_t height, uint8_t* loValue,
                           size_t loValueStride, uint8_t* hiCount,
                           size_t hiCountStride, uint8_t* hiValue,
                           size_t hiValueStride, uint8_t threshold) {
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(64)
        {
            size_t col = psim_get_thread_num();
            AdjustLo(loCount[col], loValue[col], threshold);
            AdjustHi(hiCount[col], hiValue[col], threshold);
            loCount[col] = 0;
            hiCount[col] = 0;
        }
        loValue += loValueStride;
        hiValue += hiValueStride;
        loCount += loCountStride;
        hiCount += hiCountStride;
    }
}

void BackgroundAdjustRangeMasked(uint8_t* loCount, size_t loCountStride,
                                 size_t width, size_t height, uint8_t* loValue,
                                 size_t loValueStride, uint8_t* hiCount,
                                 size_t hiCountStride, uint8_t* hiValue,
                                 size_t hiValueStride, uint8_t threshold,
                                 const uint8_t* mask, size_t maskStride) {
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(64)
        {
            size_t col = psim_get_thread_num();

            if (mask[col]) {
                AdjustLo(loCount[col], loValue[col], threshold);
                AdjustHi(hiCount[col], hiValue[col], threshold);
            }
            loCount[col] = 0;
            hiCount[col] = 0;
        }
        loValue += loValueStride;
        hiValue += hiValueStride;
        loCount += loCountStride;
        hiCount += hiCountStride;
        mask += maskStride;
    }
}

STATIC_INLINE void BackgroundShiftRange(const uint8_t& value, uint8_t& lo,
                                        uint8_t& hi) {
    uint8_t add = psim_usub_sat(value, hi);
    uint8_t sub = psim_usub_sat(lo, value);

    uint8_t lo_add = psim_uadd_sat(lo, add);
    uint8_t hi_add = psim_uadd_sat(hi, add);

    lo = psim_usub_sat(lo_add, sub);
    hi = psim_usub_sat(hi_add, sub);
}

void BackgroundShiftRange(const uint8_t* value, size_t valueStride,
                          size_t width, size_t height, uint8_t* lo,
                          size_t loStride, uint8_t* hi, size_t hiStride) {
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(64)
        {
            size_t col = psim_get_thread_num();
            BackgroundShiftRange(value[col], lo[col], hi[col]);
        }
        value += valueStride;
        lo += loStride;
        hi += hiStride;
    }
}

void BackgroundShiftRangeMasked(const uint8_t* value, size_t valueStride,
                                size_t width, size_t height, uint8_t* lo,
                                size_t loStride, uint8_t* hi, size_t hiStride,
                                const uint8_t* mask, size_t maskStride) {
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(64)
        {
            size_t col = psim_get_thread_num();
            if (mask[col]) BackgroundShiftRange(value[col], lo[col], hi[col]);
        }
        value += valueStride;
        lo += loStride;
        hi += hiStride;
        mask += maskStride;
    }
}

void BackgroundInitMask(const uint8_t* src, size_t srcStride, size_t width,
                        size_t height, uint8_t index, uint8_t value,
                        uint8_t* dst, size_t dstStride) {
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(64)
        {
            size_t col = psim_get_thread_num();
            if (src[col] == index) dst[col] = value;
        }
        src += srcStride;
        dst += dstStride;
    }
}
}  // namespace Psv
}  // namespace Simd
