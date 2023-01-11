#include <assert.h>
#include <parsim.h>
#include "Math.h"
#include "psimdlib.h"

namespace Simd {
namespace Psv {

STATIC_INLINE uint16_t AlphaBlending(uint16_t src, uint16_t dst,
                                     uint16_t alpha) {
    return DivideBy255(src * alpha + dst * (0xFF - alpha));
}

template <size_t channelCount>
void AlphaBlending(const uint8_t* src, size_t srcStride, size_t width,
                   size_t height, const uint8_t* alpha, size_t alphaStride,
                   uint8_t* dst, size_t dstStride) {
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width* channelCount) \
    gang_size(64 * channelCount)
        {
            size_t col = psim_get_thread_num();
            dst[col] =
                AlphaBlending(src[col], dst[col], alpha[col / channelCount]);
        }
        src += srcStride;
        alpha += alphaStride;
        dst += dstStride;
    }
}

void AlphaBlending(const uint8_t* src, size_t srcStride, size_t width,
                   size_t height, size_t channelCount, const uint8_t* alpha,
                   size_t alphaStride, uint8_t* dst, size_t dstStride) {
    assert(channelCount >= 1 && channelCount <= 4);

    switch (channelCount) {
        case 1:
            AlphaBlending<1>(src, srcStride, width, height, alpha, alphaStride,
                             dst, dstStride);
            break;
        case 2:
            AlphaBlending<2>(src, srcStride, width, height, alpha, alphaStride,
                             dst, dstStride);
            break;
        case 3:
            AlphaBlending<3>(src, srcStride, width, height, alpha, alphaStride,
                             dst, dstStride);
            break;
        case 4:
            AlphaBlending<4>(src, srcStride, width, height, alpha, alphaStride,
                             dst, dstStride);
            break;
    }
}

//---------------------------------------------------------------------

template <size_t channelCount>
void AlphaBlendingUniform(const uint8_t* src, size_t srcStride, size_t width,
                          size_t height, uint8_t alpha, uint8_t* dst,
                          size_t dstStride) {
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width* channelCount) gang_size(64)
        {
            size_t col = psim_get_thread_num();
            dst[col] = AlphaBlending(src[col], dst[col], alpha);
        }
        src += srcStride;
        dst += dstStride;
    }
}

void AlphaBlendingUniform(const uint8_t* src, size_t srcStride, size_t width,
                          size_t height, size_t channelCount, uint8_t alpha,
                          uint8_t* dst, size_t dstStride) {
    assert(channelCount >= 1 && channelCount <= 4);

    switch (channelCount) {
        case 1:
            AlphaBlendingUniform<1>(src, srcStride, width, height, alpha, dst,
                                    dstStride);
            break;
        case 2:
            AlphaBlendingUniform<2>(src, srcStride, width, height, alpha, dst,
                                    dstStride);
            break;
        case 3:
            AlphaBlendingUniform<3>(src, srcStride, width, height, alpha, dst,
                                    dstStride);
            break;
        case 4:
            AlphaBlendingUniform<4>(src, srcStride, width, height, alpha, dst,
                                    dstStride);
            break;
    }
}

//---------------------------------------------------------------------

template <size_t channelCount>
void AlphaFilling(uint8_t* dst, size_t dstStride, size_t width, size_t height,
                  const uint8_t* channel, const uint8_t* alpha,
                  size_t alphaStride) {
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width* channelCount) \
    gang_size(128 * channelCount)
        {
            size_t col = psim_get_thread_num();
            dst[col] = AlphaBlending(channel[col % channelCount], dst[col],
                                     alpha[col / channelCount]);
        }
        alpha += alphaStride;
        dst += dstStride;
    }
}

void AlphaFilling(uint8_t* dst, size_t dstStride, size_t width, size_t height,
                  const uint8_t* channel, size_t channelCount,
                  const uint8_t* alpha, size_t alphaStride) {
    assert(channelCount >= 1 && channelCount <= 4);

    switch (channelCount) {
        case 1:
            AlphaFilling<1>(dst, dstStride, width, height, channel, alpha,
                            alphaStride);
            break;
        case 2:
            AlphaFilling<2>(dst, dstStride, width, height, channel, alpha,
                            alphaStride);
            break;
        case 3:
            AlphaFilling<3>(dst, dstStride, width, height, channel, alpha,
                            alphaStride);
            break;
        case 4:
            AlphaFilling<4>(dst, dstStride, width, height, channel, alpha,
                            alphaStride);
            break;
    }
}

//---------------------------------------------------------------------

#if 0

void AlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width,
                      size_t height, uint8_t* dst, size_t dstStride) {
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width * 4) gang_size(64)
        {
            size_t col = psim_get_thread_num();
            size_t alpha_id = ((col / 4 + 1) * 4) - 1;
            uint8_t alpha = src[alpha_id];
            uint8_t final = (col & (col >> 1) & 1) * 0xFF | alpha;
            dst[col] = DivideBy255(src[col] * final);
        }
        src += srcStride;
        dst += dstStride;
    }
}

void AlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width,
                      size_t height, uint8_t* dst, size_t dstStride) {
    for (size_t row = 0; row < height; ++row) {
        size_t ws = 64;
        size_t ncols = roundUp(width * 4, ws);
#psim num_spmd_threads(ncols) gang_size(ws)
        {
            size_t col = psim_get_thread_num();
            uint32_t lane = psim_get_lane_num();
            uint8_t val = 0;
            if (!psim_is_tail_gang() || col < width * 4) {
                val = src[col];
            }
            uint32_t shf_ptrn = 3 + 4 * (lane / 4);
            uint8_t alpha = psim_shfl_sync<uint8_t>(val, shf_ptrn);
            uint8_t final = (col & (col >> 1) & 1) * 0xFF | alpha;
            if (!psim_is_tail_gang() || col < width * 4) {
                dst[col] = DivideBy255(src[col] * final);
            }
        }
        src += srcStride;
        dst += dstStride;
    }
}

#else

void AlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width,
                      size_t height, uint8_t* dst, size_t dstStride) {
	size_t ws = 64;
	size_t ncols = roundUp(width * 4, ws);
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(ncols) gang_size(ws)
        {
            size_t col = psim_get_thread_num();
            uint8_t val = 0;
            if (!psim_is_tail_gang() || col < width * 4) {
                val = src[col];
            }
            uint32_t val_32 = psim_zip_sync<uint32_t>(val);
            uint32_t b = (uint8_t)val_32;
            uint32_t g = (uint8_t)(val_32 >> 8);
            uint32_t r = (uint8_t)(val_32 >> 16);
            uint32_t a = (uint8_t)(val_32 >> 24);

            b = DivideBy255(b * a);
            g = DivideBy255(g * a) << 8;
            r = DivideBy255(r * a) << 16;
            a = a << 24;
            uint32_t out_32 = b | g | r | a;
            uint8_t out = psim_unzip_sync<uint8_t>(out_32, 0);
            if (!psim_is_tail_gang() || col < width * 4) {
                dst[col] = out;
            }
        }
        src += srcStride;
        dst += dstStride;
    }
}

#endif
//---------------------------------------------------------------------

#if 0

void AlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width,
                        size_t height, uint8_t* dst, size_t dstStride) {
    for (size_t row = 0; row < height; ++row) {
        size_t ws = 64;
#psim num_spmd_threads(width) gang_size(ws)
        {
            size_t col = psim_get_thread_num();
            size_t alpha_id = ((col / 4 + 1) * 4) - 1;
            float alpha = src[alpha_id];

            float alpha_inv = alpha ? 255.00001f / alpha : 0.0f;
            float final_alpha = (col & (col >> 1) & 1) ? 1.0f : alpha_inv;
            dst[col] = RestrictRange(int(src[col] * final_alpha));
        }
        src += srcStride;
        dst += dstStride;
    }
}

#else

void AlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width,
                        size_t height, uint8_t* dst, size_t dstStride) {
	size_t ws = 64;
	size_t ncols = roundUp(width * 4, ws);
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(ncols) gang_size(ws)
        {
            size_t col = psim_get_thread_num();
            uint8_t val = 0;
            if (!psim_is_tail_gang() || col < width * 4) {
                val = src[col];
            }
            uint32_t val_32 = psim_zip_sync<uint32_t>(val);
            uint32_t b = (uint8_t)val_32;
            uint32_t g = (uint8_t)(val_32 >> 8);
            uint32_t r = (uint8_t)(val_32 >> 16);
            uint32_t a = (uint8_t)(val_32 >> 24);
            float alpha = a ? 255.00001f / a : 0.0f;
            b = RestrictRange(int(b * alpha));
            g = RestrictRange(int(g * alpha)) << 8;
            r = RestrictRange(int(r * alpha)) << 16;
            a = a << 24;
            uint32_t out_32 = b | g | r | a;
            uint8_t out = psim_unzip_sync<uint8_t>(out_32, 0);
            if (!psim_is_tail_gang() || col < width * 4) {
                dst[col] = out;
            }
        }
        src += srcStride;
        dst += dstStride;
    }
}

#endif
}  // namespace Psv
}  // namespace Simd
