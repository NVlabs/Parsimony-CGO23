#include <parsim.h>
#include "Math.h"
#include "psimdlib.h"

namespace Simd {
namespace Psv {

void AbsDifference(const uint8_t* a, size_t aStride, const uint8_t* b,
                   size_t bStride, uint8_t* c, size_t cStride, size_t width,
                   size_t height) {
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(64)
        {
            int col = psim_get_lane_num() +
                      psim_get_gang_num() * psim_get_gang_size();
            c[col] = AbsDiff(a[col], b[col]);
        }
        a += aStride;
        b += bStride;
        c += cStride;
    }
}
}  // namespace Psv
}  // namespace Simd
