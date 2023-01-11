#include <parsim.h>
#include "psimdlib.h"

namespace Simd {
namespace Psv {

void NeuralConvert(const uint8_t* src, size_t srcStride, size_t width,
                   size_t height, float* dst, size_t dstStride, int inversion) {
    const float k = 1.0f / 255.0f;
    if (inversion) {
        for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(16)
            {
                size_t col = psim_get_thread_num();
                dst[col] = (255 - src[col]) * k;
            }
            src += srcStride;
            dst += dstStride;
        }
    } else {
        for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(16)
            {
                size_t col = psim_get_thread_num();
                dst[col] = src[col] * k;
            }
            src += srcStride;
            dst += dstStride;
        }
    }
}

}  // namespace Psv
}  // namespace Simd
