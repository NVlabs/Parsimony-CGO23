#include <parsim.h>
#include <cstring>
#include "Math.h"
#include "psimdlib.h"

namespace Simd {
namespace Psv {

void AbsGradientSaturatedSum(const uint8_t* src, size_t srcStride, size_t width,
                             size_t height, uint8_t* dst, size_t dstStride) {
    memset(dst, 0, width);
    src += srcStride;
    dst += dstStride;
    for (size_t row = 2; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(64)
        {
            size_t col = psim_get_lane_num() +
                         psim_get_gang_num() * psim_get_gang_size();
            const uint8_t dy =
                AbsDiff(src[col - srcStride], src[col + srcStride]);
            const uint8_t dx = AbsDiff(src[col - 1], src[col + 1]);
            dst[col] = psim_uadd_sat(dx, dy);
        }
        dst[0] = 0;
        dst[width - 1] = 0;

        src += srcStride;
        dst += dstStride;
    }
    memset(dst, 0, width);
}

}  // namespace Psv
}  // namespace Simd
