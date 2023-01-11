#include <assert.h>
#include <parsim.h>
#include "Math.h"
#include "psimdlib.h"

namespace Simd {
namespace Psv {

void AbsDifferenceSum(const uint8_t* a, size_t aStride, const uint8_t* b,
                      size_t bStride, size_t width, size_t height,
                      uint64_t* sum) {
    const int GS = 256;
    PsimCollectiveAddAbsDiff<uint64_t> _sum;
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(GS)
        {
            size_t col = psim_get_thread_num();
            _sum.AddAbsDiff(a[col], b[col]);
        }

        a += aStride;
        b += bStride;
    }

    *sum = _sum.ReduceSum();
}

void AbsDifferenceSumMasked(const uint8_t* a, size_t aStride, const uint8_t* b,
                            size_t bStride, const uint8_t* mask,
                            size_t maskStride, uint8_t index, size_t width,
                            size_t height, uint64_t* sum) {
    const int GS = 256;
    PsimCollectiveAddAbsDiff<uint64_t> _sum;
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(GS)
        {
            size_t col = psim_get_thread_num();
            if (mask[col] == index) {
                _sum.AddAbsDiff(a[col], b[col]);
            }
        }

        a += aStride;
        b += bStride;
        mask += maskStride;
    }

    *sum = _sum.ReduceSum();
}

void AbsDifferenceSums3x3(const uint8_t* current, size_t currentStride,
                          const uint8_t* background, size_t backgroundStride,
                          size_t width, size_t height, uint64_t* sums) {
    assert(width > 2 && height > 2);

    height -= 2;
    width -= 2;
    current += 1 + currentStride;
    background += 1 + backgroundStride;

    const int GS = 256;
    PsimCollectiveAddAbsDiff<uint64_t> _sums[9];
    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(GS)
        {
            size_t col = psim_get_thread_num();
            uint8_t value = current[col];

            _sums[0].AddAbsDiff(value, background[col - backgroundStride - 1]);
            _sums[1].AddAbsDiff(value, background[col - backgroundStride]);
            _sums[2].AddAbsDiff(value, background[col - backgroundStride + 1]);
            _sums[3].AddAbsDiff(value, background[col - 1]);
            _sums[4].AddAbsDiff(value, background[col]);
            _sums[5].AddAbsDiff(value, background[col + 1]);
            _sums[6].AddAbsDiff(value, background[col + backgroundStride - 1]);
            _sums[7].AddAbsDiff(value, background[col + backgroundStride]);
            _sums[8].AddAbsDiff(value, background[col + backgroundStride + 1]);
        }

        current += currentStride;
        background += backgroundStride;
    }

    for (size_t i = 0; i < 9; ++i) {
        sums[i] = _sums[i].ReduceSum();
    }
}

void AbsDifferenceSums3x3Masked(const uint8_t* current, size_t currentStride,
                                const uint8_t* background,
                                size_t backgroundStride, const uint8_t* mask,
                                size_t maskStride, uint8_t index, size_t width,
                                size_t height, uint64_t* sums) {
    assert(width > 2 && height > 2);

    height -= 2;
    width -= 2;
    current += 1 + currentStride;
    background += 1 + backgroundStride;
    mask += 1 + maskStride;

    const int GS = 256;
    PsimCollectiveAddAbsDiff<uint64_t> _sums[9];

    for (size_t row = 0; row < height; ++row) {
#psim num_spmd_threads(width) gang_size(GS)
        {
            size_t col = psim_get_thread_num();
            if (mask[col] == index) {
                uint8_t value = current[col];
                _sums[0].AddAbsDiff(value,
                                    background[col - backgroundStride - 1]);
                _sums[1].AddAbsDiff(value, background[col - backgroundStride]);
                _sums[2].AddAbsDiff(value,
                                    background[col - backgroundStride + 1]);
                _sums[3].AddAbsDiff(value, background[col - 1]);
                _sums[4].AddAbsDiff(value, background[col]);
                _sums[5].AddAbsDiff(value, background[col + 1]);
                _sums[6].AddAbsDiff(value,
                                    background[col + backgroundStride - 1]);
                _sums[7].AddAbsDiff(value, background[col + backgroundStride]);
                _sums[8].AddAbsDiff(value,
                                    background[col + backgroundStride + 1]);
            }
        }

        current += currentStride;
        background += backgroundStride;
        mask += maskStride;
    }

    for (size_t i = 0; i < 9; ++i) {
        sums[i] = _sums[i].ReduceSum();
    }
}

}  // namespace Psv
}  // namespace Simd
