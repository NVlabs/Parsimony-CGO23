#pragma once
#include <cstdint>
#include <cstdio>

// include for SimdPixelFormatType
#include "../src/Simd/SimdLib.h"

namespace Simd {
namespace Psv {

/* AbsDifference.cpp */
void AbsDifference(const uint8_t* a, size_t aStride, const uint8_t* b,
                   size_t bStride, uint8_t* c, size_t cStride, size_t width,
                   size_t height);

/* AbsDifferenceSum.cpp */
void AbsDifferenceSum(const uint8_t* a, size_t aStride, const uint8_t* b,
                      size_t bStride, size_t width, size_t height,
                      uint64_t* sum);

void AbsDifferenceSumMasked(const uint8_t* a, size_t aStride, const uint8_t* b,
                            size_t bStride, const uint8_t* mask,
                            size_t maskStride, uint8_t index, size_t width,
                            size_t height, uint64_t* sum);

void AbsDifferenceSums3x3(const uint8_t* current, size_t currentStride,
                          const uint8_t* background, size_t backgroundStride,
                          size_t width, size_t height, uint64_t* sums);

void AbsDifferenceSums3x3Masked(const uint8_t* current, size_t currentStride,
                                const uint8_t* background,
                                size_t backgroundStride, const uint8_t* mask,
                                size_t maskStride, uint8_t index, size_t width,
                                size_t height, uint64_t* sums);

/* AbsGradientSaturatedSum.cpp */
void AbsGradientSaturatedSum(const uint8_t* src, size_t srcStride, size_t width,
                             size_t height, uint8_t* dst, size_t dstStride);

/* AddFeatureDifference.cpp */
void AddFeatureDifference(const uint8_t* value, size_t valueStride,
                          size_t width, size_t height, const uint8_t* lo,
                          size_t loStride, const uint8_t* hi, size_t hiStride,
                          uint16_t weight, uint8_t* difference,
                          size_t differenceStride);

/* AlphaBlending.cpp */
void AlphaBlending(const uint8_t* src, size_t srcStride, size_t width,
                   size_t height, size_t channelCount, const uint8_t* alpha,
                   size_t alphaStride, uint8_t* dst, size_t dstStride);

void AlphaBlendingUniform(const uint8_t* src, size_t srcStride, size_t width,
                          size_t height, size_t channelCount, uint8_t alpha,
                          uint8_t* dst, size_t dstStride);

void AlphaFilling(uint8_t* dst, size_t dstStride, size_t width, size_t height,
                  const uint8_t* channel, size_t channelCount,
                  const uint8_t* alpha, size_t alphaStride);

void AlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width,
                      size_t height, uint8_t* dst, size_t dstStride);

void AlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width,
                        size_t height, uint8_t* dst, size_t dstStride);

/* BaseBackground.cpp */
void BackgroundGrowRangeSlow(const uint8_t* value, size_t valueStride,
                             size_t width, size_t height, uint8_t* lo,
                             size_t loStride, uint8_t* hi, size_t hiStride);

void BackgroundGrowRangeFast(const uint8_t* value, size_t valueStride,
                             size_t width, size_t height, uint8_t* lo,
                             size_t loStride, uint8_t* hi, size_t hiStride);

void BackgroundIncrementCount(const uint8_t* value, size_t valueStride,
                              size_t width, size_t height,
                              const uint8_t* loValue, size_t loValueStride,
                              const uint8_t* hiValue, size_t hiValueStride,
                              uint8_t* loCount, size_t loCountStride,
                              uint8_t* hiCount, size_t hiCountStride);

void BackgroundAdjustRange(uint8_t* loCount, size_t loCountStride, size_t width,
                           size_t height, uint8_t* loValue,
                           size_t loValueStride, uint8_t* hiCount,
                           size_t hiCountStride, uint8_t* hiValue,
                           size_t hiValueStride, uint8_t threshold);

void BackgroundAdjustRangeMasked(uint8_t* loCount, size_t loCountStride,
                                 size_t width, size_t height, uint8_t* loValue,
                                 size_t loValueStride, uint8_t* hiCount,
                                 size_t hiCountStride, uint8_t* hiValue,
                                 size_t hiValueStride, uint8_t threshold,
                                 const uint8_t* mask, size_t maskStride);

void BackgroundShiftRange(const uint8_t* value, size_t valueStride,
                          size_t width, size_t height, uint8_t* lo,
                          size_t loStride, uint8_t* hi, size_t hiStride);

void BackgroundShiftRangeMasked(const uint8_t* value, size_t valueStride,
                                size_t width, size_t height, uint8_t* lo,
                                size_t loStride, uint8_t* hi, size_t hiStride,
                                const uint8_t* mask, size_t maskStride);

void BackgroundInitMask(const uint8_t* src, size_t srcStride, size_t width,
                        size_t height, uint8_t index, uint8_t value,
                        uint8_t* dst, size_t dstStride);

/* Base64.cpp */
void Base64Decode(const uint8_t* src, size_t srcSize, uint8_t* dst,
                  size_t* dstSize);
void Base64Encode(const uint8_t* src, size_t size, uint8_t* dst);

/* BayerToBrg.cpp */
void BayerToBgr(const uint8_t* bayer, size_t width, size_t height,
                size_t bayerStride, SimdPixelFormatType bayerFormat,
                uint8_t* bgr, size_t bgrStride);

void BayerToBgra(const uint8_t* bayer, size_t width, size_t height,
                 size_t bayerStride, SimdPixelFormatType bayerFormat,
                 uint8_t* bgra, size_t bgraStride, uint8_t alpha);

/* BgrToBaryer.cpp */
void BgrToBayer(const uint8_t* bgr, size_t width, size_t height,
                size_t bgrStride, uint8_t* bayer, size_t bayerStride,
                SimdPixelFormatType bayerFormat);

/* Binarization.cpp */
void Binarization(const uint8_t* src, size_t srcStride, size_t width,
                  size_t height, uint8_t value, uint8_t positive,
                  uint8_t negative, uint8_t* dst, size_t dstStride,
                  SimdCompareType compareType);

void AveragingBinarization(const uint8_t* src, size_t srcStride, size_t width,
                           size_t height, uint8_t value, size_t neighborhood,
                           uint8_t threshold, uint8_t positive,
                           uint8_t negative, uint8_t* dst, size_t dstStride,
                           SimdCompareType compareType);

/* Conditional.cpp */
void ConditionalFill(const uint8_t * src, size_t srcStride, size_t width, size_t height,
    uint8_t threshold, SimdCompareType compareType, uint8_t value, uint8_t * dst, size_t dstStride);

void ConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
    const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum);

void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height,
    int16_t value, SimdCompareType compareType, uint32_t * count);

void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height,
    uint8_t value, SimdCompareType compareType, uint32_t * count);

/* Neural.cpp */
void NeuralConvert(const uint8_t* src, size_t srcStride, size_t width,
                   size_t height, float* dst, size_t dstStride, int inversion);

}  // namespace Psv
}  // namespace Simd
