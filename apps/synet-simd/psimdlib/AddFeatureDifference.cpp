#include <parsim.h>
#include <algorithm>
#include <cstring>
#include "Math.h"
#include "psimdlib.h"

namespace Simd {
namespace Psv {

STATIC_INLINE uint16_t ShiftedWeightedSquare(uint8_t difference,
                                             uint16_t weight) {
    return psim_umulh(difference * difference, weight);
}

STATIC_INLINE uint8_t FeatureDifference(uint8_t value, uint8_t lo, uint8_t hi) {
    uint8_t val1 = psim_usub_sat(value, hi);
    uint8_t val2 = psim_usub_sat(lo, value);
    return std::max(val1, val2);
}

void AddFeatureDifference(const uint8_t* value, size_t valueStride,
                          size_t width, size_t height, const uint8_t* lo,
                          size_t loStride, const uint8_t* hi, size_t hiStride,
                          uint16_t weight, uint8_t* difference,
                          size_t differenceStride) {
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(64)
        {
            size_t col = psim_get_thread_num();

            uint8_t featureDifference =
                FeatureDifference(value[col], lo[col], hi[col]);

            uint32_t sum = difference[col] +
                           ShiftedWeightedSquare(featureDifference, weight);
            difference[col] = std::min((int)sum, (int)0xFF);
        }

        value += valueStride;
        lo += loStride;
        hi += hiStride;
        difference += differenceStride;
    }
}

}  // namespace Psv
}  // namespace Simd
