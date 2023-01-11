#pragma once
#include <algorithm>
// include for SimdCompareType
#include "../src/Simd/SimdLib.h"

#define STATIC_INLINE inline __attribute__((always_inline))

template <typename T>
STATIC_INLINE T AbsDiff(T a, T b) {
    return std::max(a, b) - std::min(a, b);
}

STATIC_INLINE int RestrictRange(int value, int min = 0, int max = 255) {
    return std::max(min, std::min(max, value));
}

template <typename T>
STATIC_INLINE T roundUp(T a, T b) {
    return ((a + (b - 1)) / b) * b;
}

template <typename T>
STATIC_INLINE T CeilDiv(T a, T b) {
    return ((a + (b - 1)) / b);
}

STATIC_INLINE size_t AlignLoAny(size_t size, size_t align) {
    return size / align * align;
}

template <typename T>
STATIC_INLINE T Average(T a, T b) {
    return (a + b + 1) >> 1;
}

template <typename T>
STATIC_INLINE T Average(T a, T b, T c, T d) {
    return (a + b + c + d + 2) >> 2;
}

STATIC_INLINE uint16_t DivideBy255(uint16_t value) {
#if 0
    // Naive version of DivideBy255
    return (value + 1 + (value >> 8)) >> 8;
#else
    return psim_umulh(value + 1, 257);
#endif
}

template <SimdCompareType type>
STATIC_INLINE bool Compare8u(const uint8_t& src, const uint8_t& b);

template <>
STATIC_INLINE bool Compare8u<SimdCompareEqual>(const uint8_t& a,
                                               const uint8_t& b) {
    return a == b;
}

template <>
STATIC_INLINE bool Compare8u<SimdCompareNotEqual>(const uint8_t& a,
                                                  const uint8_t& b) {
    return a != b;
}

template <>
STATIC_INLINE bool Compare8u<SimdCompareGreater>(const uint8_t& a,
                                                 const uint8_t& b) {
    return a > b;
}

template <>
STATIC_INLINE bool Compare8u<SimdCompareGreaterOrEqual>(const uint8_t& a,
                                                        const uint8_t& b) {
    return a >= b;
}

template <>
STATIC_INLINE bool Compare8u<SimdCompareLesser>(const uint8_t& a,
                                                const uint8_t& b) {
    return a < b;
}

template <>
STATIC_INLINE bool Compare8u<SimdCompareLesserOrEqual>(const uint8_t& a,
                                                       const uint8_t& b) {
    return a <= b;
}

template <SimdCompareType type> STATIC_INLINE bool Compare16i(const int16_t & src, const int16_t & b);

template <> STATIC_INLINE bool Compare16i<SimdCompareEqual>(const int16_t & a, const int16_t & b)
{
    return a == b;
}

template <> STATIC_INLINE bool Compare16i<SimdCompareNotEqual>(const int16_t & a, const int16_t & b)
{
    return a != b;
}

template <> STATIC_INLINE bool Compare16i<SimdCompareGreater>(const int16_t & a, const int16_t & b)
{
    return a > b;
}

template <> STATIC_INLINE bool Compare16i<SimdCompareGreaterOrEqual>(const int16_t & a, const int16_t & b)
{
    return a >= b;
}

template <> STATIC_INLINE bool Compare16i<SimdCompareLesser>(const int16_t & a, const int16_t & b)
{
    return a < b;
}

template <> STATIC_INLINE bool Compare16i<SimdCompareLesserOrEqual>(const int16_t & a, const int16_t & b)
{
    return a <= b;
}

STATIC_INLINE int Square(int a)
{
    return a*a;
}

STATIC_INLINE int SquaredDifference(int a, int b)
{
    return Square(a - b);
}