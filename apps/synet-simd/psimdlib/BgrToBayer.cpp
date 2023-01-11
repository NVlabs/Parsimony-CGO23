#include <assert.h>
#include <parsim.h>
#include "Math.h"
#include "psimdlib.h"

namespace Simd {
namespace Psv {

#define GS 64

int PTRN[4][2][2] = {
    {{1, 2}, {0, 1}}, {{1, 0}, {2, 1}}, {{2, 1}, {1, 0}}, {{0, 1}, {1, 2}}};

template <int format, int row>
STATIC_INLINE void BgrToBayer(const uint8_t* bgr, size_t width,
                              uint8_t* bayer) {
    size_t num_gangs = CeilDiv(width, (size_t)GS);
#psim num_spmd_gangs(num_gangs) gang_size(GS)
    {
        size_t gang = psim_get_gang_num();
        size_t lane = psim_get_lane_num();

        const uint8_t* pbgr = &(bgr[3 * (gang * GS + lane)]);
        int ptrn = PTRN[format][row][lane % 2];

        bayer[gang * GS + lane] = pbgr[ptrn];
    }
}

template <int format>
void BgrToBayer(const uint8_t* bgr, size_t width, size_t height,
                size_t bgrStride, uint8_t* bayer, size_t bayerStride) {
    for (size_t row = 0; row < height; row += 2) {
        BgrToBayer<format, 0>(bgr, width, bayer);
        bgr += bgrStride;
        bayer += bayerStride;

        BgrToBayer<format, 1>(bgr, width, bayer);
        bgr += bgrStride;
        bayer += bayerStride;
    }
}

void BgrToBayer(const uint8_t* bgr, size_t width, size_t height,
                size_t bgrStride, uint8_t* bayer, size_t bayerStride,
                SimdPixelFormatType bayerFormat) {
    assert((width % 2 == 0) && (height % 2 == 0));
    switch (bayerFormat) {
        case SimdPixelFormatBayerGrbg:
            BgrToBayer<0>(bgr, width, height, bgrStride, bayer, bayerStride);
            break;
        case SimdPixelFormatBayerGbrg:
            BgrToBayer<1>(bgr, width, height, bgrStride, bayer, bayerStride);
            break;
        case SimdPixelFormatBayerRggb:
            BgrToBayer<2>(bgr, width, height, bgrStride, bayer, bayerStride);
            break;
        case SimdPixelFormatBayerBggr:
            BgrToBayer<3>(bgr, width, height, bgrStride, bayer, bayerStride);
            break;
        default:
            assert(0);
    }
}

}  // namespace Psv
}  // namespace Simd
