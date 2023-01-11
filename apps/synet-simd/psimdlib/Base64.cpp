#include <assert.h>
#include <parsim.h>
#include "Math.h"
#include "psimdlib.h"

namespace Simd {
namespace Psv {

#define WS 64

#ifdef PSV_SIMPLE

void Base64Encode(const uint8_t* src, size_t size, uint8_t* dst) {
    size_t size3 = AlignLoAny(size, 3);
    size_t size_dst = size3 / 3 * 4;
    if (size > size3) size_dst += 4;

    size_t num_gangs = CeilDiv(size_dst, (size_t)WS);
#psim num_spmd_gangs(num_gangs) gang_size(WS)
    {
        size_t lane_id = psim_get_lane_num();
        size_t src_elem_id = 48 * psim_get_gang_num() + lane_id;
        size_t dst_elem_id = psim_get_thread_num();
        uint8_t my_src = 0;
        uint8_t my_dst = 0;
        if (psim_is_tail_gang()) {
            if (src_elem_id < size) {
                my_src = src[src_elem_id];
            }
        } else {
            my_src = src[src_elem_id];
        }

        uint8_t shuffle_id0 = lane_id - (lane_id / 4);
        uint8_t shuffle_id1 = lane_id - 1 < 0 ? 0 : (lane_id - 1);
        my_src = psim_shuffle_sync<uint8_t>(my_src, shuffle_id0);
        uint8_t prev_src = psim_shuffle_sync<uint8_t>(my_src, shuffle_id1);
        uint8_t index = 0;

        if (lane_id % 4 == 0)
            index = (my_src & 0xfc) >> 2;
        else if (lane_id % 4 == 1)
            index = ((prev_src & 0x03) << 4) | ((my_src & 0xf0) >> 4);
        else if (lane_id % 4 == 2)
            index = ((prev_src & 0x0f) << 2) | ((my_src & 0xc0) >> 6);
        else if (lane_id % 4 == 3)
            index = prev_src & 0x3f;

        bool isUpCase = index < 26;
        bool isLetter = index < 52;
        bool isDigit = index < 62;

        uint8_t upValue = isUpCase ? 'A' + index : 0;
        uint8_t lowValue = (isLetter & !isUpCase) ? 'a' + index - 26 : 0;
        uint8_t digValue = (isDigit & !isLetter) ? '0' + index - 52 : 0;
        uint8_t sign_p = index == 62 ? '+' : 0;
        uint8_t sign_d = index == 63 ? '/' : 0;
        my_dst = upValue | lowValue | digValue | sign_p | sign_d;

        if (psim_is_tail_gang()) {
            if (dst_elem_id < size_dst) {
                dst[dst_elem_id] = my_dst;
                if (size - size3) {
                    if (dst_elem_id == size_dst - 1) {
                        dst[dst_elem_id] = '=';
                    }
                }
                if (size - size3 == 1) {
                    if (dst_elem_id == size_dst - 2) {
                        dst[dst_elem_id] = '=';
                    }
                }
            }
        } else {
            dst[dst_elem_id] = my_dst;
        }
    }
}

void Base64Decode(const uint8_t* src, size_t srcSize, uint8_t* dst,
                  size_t* dstSize) {
    assert(srcSize % 4 == 0 && srcSize >= 4);
    size_t srcSize4 = srcSize - 2;
    size_t dstSize4 = (srcSize / 4 * 3);
    size_t tailElems = 1;
    if (src[srcSize - 2] != '=') {
        srcSize4++;
        tailElems++;
        if (src[srcSize - 1] != '=') {
            srcSize4++;
            tailElems++;
        }
    }

    size_t num_gangs = CeilDiv(srcSize4, (size_t)WS);
#psim num_spmd_gangs(num_gangs) gang_size(WS)
    {
        uint32_t lane_id = psim_get_lane_num();
        size_t src_elem_id = psim_get_thread_num();
        size_t dst_elem_id = 48 * psim_get_gang_num() + lane_id;
        uint8_t my_src = 0;
        uint8_t my_dst = 0;

        if (psim_is_tail_gang()) {
            if (src_elem_id < srcSize4) {
                my_src = src[src_elem_id];
            }
        } else {
            my_src = src[src_elem_id];
        }

        bool isLetter = my_src > 'Z';
        bool isUpCase = my_src > '9';
        bool isDigit = my_src > '/';

        uint8_t upValue = isLetter ? my_src - 71 : 0;
        uint8_t lowValue = (isUpCase & !isLetter) ? my_src - 'A' : 0;
        uint8_t digValue = (isDigit & !isUpCase) ? my_src + 4 : 0;
        uint8_t sign_p = my_src == '+' ? 62 : 0;
        uint8_t sign_d = my_src == '/' ? 63 : 0;
        uint32_t from = upValue | lowValue | digValue | sign_p | sign_d;

        uint8_t shfl_idx = lane_id / 3 * 4;
        uint32_t s0 = psim_shuffle_sync<uint32_t>(from, shfl_idx);
        uint32_t s1 = psim_shuffle_sync<uint32_t>(from, shfl_idx + 1);
        uint32_t s2 = psim_shuffle_sync<uint32_t>(from, shfl_idx + 2);
        uint32_t s3 = psim_shuffle_sync<uint32_t>(from, shfl_idx + 3);
        uint32_t n = s0 << 18 | s1 << 12 | s2 << 6 | s3;
        if (lane_id % 3 == 0)
            my_dst = n >> 16;
        else if (lane_id % 3 == 1)
            my_dst = n >> 8 & 0xFF;
        else if (lane_id % 3 == 2)
            my_dst = n & 0xFF;

        if (psim_is_tail_gang()) {
            if (dst_elem_id < dstSize4) {
                dst[dst_elem_id] = my_dst;
            }
        } else {
            if (lane_id < 48) {
                dst[dst_elem_id] = my_dst;
            }
        }
    }
    *dstSize = dstSize4 - 3 + tailElems;
}

#else

/* clang-format off */

#define PATTERN_DECODE8_LO \
    {1, 3,  2,  5,  7,  6,  9,  11, 10, 13, 15, 14, -1, -1, -1, -1, \
    17, 19, 18, 21, 23, 22, 25, 27, 26, 29, 31, 30, -1, -1, -1, -1, \
    33, 35, 34, 37, 39, 38, 41, 43, 42, 45, 47, 46, -1, -1, -1, -1, \
    49, 51, 50, 53, 55, 54, 57, 59, 58, 61, 63, 62, -1, -1, -1, -1 }

#define PATTERN_DECODE8_HI \
    {1, 0,  2,  5,  4,  6,  9,  8,  10, 13, 12, 14, -1, -1, -1, -1, \
    17, 16, 18, 21, 20, 22, 25, 24, 26, 29, 28, 30, -1, -1, -1, -1, \
    33, 32, 34, 37, 36, 38, 41, 40, 42, 45, 44, 46, -1, -1, -1, -1, \
    49, 48, 50, 53, 52, 54, 57, 56, 58, 61, 60, 62, -1, -1, -1, -1 }


#define PATTERN_DECODE32 \
    {0, 1,  2,  4,  5,  6,  8,  9,  10, 12, 13, 14, 0,  0,  0,  0,  \
     16,  17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 16, 16, 16, 16, \
     32,  33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 32, 32, 32, 32, \
     48,  49, 50, 52, 53, 54, 56, 57, 58, 60, 61, 62, 48, 48, 48, 48, }

#if 0
#define PATTERNLo                                                           \
    {                                                                       \
        0x1, 0x3, 0x2, 0x5, 0x7, 0x6, 0x9, 0xB, 0xA, 0xD, 0xF, 0xE, -1, -1, \
            -1, -1                                                          \
    }
#define PATTERNHi                                                           \
    {                                                                       \
        0x1, 0x0, 0x2, 0x5, 0x4, 0x6, 0x9, 0x8, 0xA, 0xD, 0xC, 0xE, -1, -1, \
            -1, -1                                                          \
    }
#endif
/* clang-format on */

void Base64Decode(const uint8_t* src, size_t srcSize, uint8_t* dst,
                  size_t* dstSize) {
    assert(srcSize % 4 == 0 && srcSize >= 4);
    size_t srcSize4 = srcSize - 2;
    size_t dstSize4 = (srcSize / 4 * 3);
    size_t tailElems = 1;
    if (src[srcSize - 2] != '=') {
        srcSize4++;
        tailElems++;
        if (src[srcSize - 1] != '=') {
            srcSize4++;
            tailElems++;
        }
    }

    size_t num_gangs = CeilDiv(srcSize4, (size_t)WS);
#psim num_spmd_gangs(num_gangs) gang_size(WS)
    {
        uint32_t lane_id = psim_get_lane_num();
        size_t src_elem_id = psim_get_thread_num();
        size_t dst_elem_id = 48 * psim_get_gang_num() + lane_id;
        uint8_t my_src = 0;
        uint8_t my_dst = 0;

        if (psim_is_tail_gang()) {
            if (src_elem_id < srcSize4) {
                my_src = src[src_elem_id];
            }
        } else {
            my_src = src[src_elem_id];
        }

        bool isLetter = my_src > 'Z';
        bool isUpCase = my_src > '9';
        bool isDigit = my_src > '/';

        uint8_t upValue = isLetter ? my_src - 71 : 0;
        uint8_t lowValue = (isUpCase & !isLetter) ? my_src - 'A' : 0;
        uint8_t digValue = (isDigit & !isUpCase) ? my_src + 4 : 0;
        uint8_t sign_p = my_src == '+' ? 62 : 0;
        uint8_t sign_d = my_src == '/' ? 63 : 0;
        uint8_t from = upValue | lowValue | digValue | sign_p | sign_d;

        uint16_t from16 = psim_zip_sync<uint16_t>(from);
        uint16_t from16_lo = from16 & 0x003F;
        uint16_t from16_hi = from16 & 0x3F00;
        uint16_t mullo_op = (lane_id % 2 == 0) ? 0x0400 : 0x0040;
        uint16_t mullo = from16_lo * mullo_op;
        uint16_t mulhi_op = (lane_id % 2 == 0) ? 0x1000 : 0x0100;
        uint16_t mulhi = psim_umulh(from16_hi, mulhi_op);

        const int shflLo[] = PATTERN_DECODE8_LO;
        const int shflHi[] = PATTERN_DECODE8_HI;
        const int shfl32[] = PATTERN_DECODE32;

        uint8_t shuffleLo = psim_unzip_sync<uint8_t>(mullo, 0);
        uint8_t shuffleHi = psim_unzip_sync<uint8_t>(mulhi, 0);

        // uint32_t shfl_ptrn_lo = shflLo[lane_id % 16];
        // if (shfl_ptrn_lo != -1) {
        //    shfl_ptrn_lo += lane_id / 16 * 16;
        //}

        // shuffleLo = psim_shuffle_sync<uint8_t>(
        //    shuffleLo, shflLo[lane_id % 16] + lane_id / 16 * 16);
        // shuffleHi = psim_shuffle_sync<uint8_t>(shuffleHi, sh
        //    shuffleHi, shflHi[lane_id % 16] + lane_id / 16 * 16);
        shuffleLo = psim_shuffle_sync<uint8_t>(shuffleLo, shflLo[lane_id]);
        shuffleHi = psim_shuffle_sync<uint8_t>(shuffleHi, shflHi[lane_id]);

        uint8_t shuffle = shuffleLo | shuffleHi;

        uint32_t permute32 = psim_zip_sync<uint32_t>(shuffle);

        permute32 = psim_shuffle_sync<uint32_t>(permute32, shfl32[lane_id]);
        my_dst = psim_unzip_sync<uint8_t>(permute32, 0);

        if (psim_is_tail_gang()) {
            if (dst_elem_id < dstSize4) {
                dst[dst_elem_id] = my_dst;
            }
        } else {
            if (lane_id < 48) {
                dst[dst_elem_id] = my_dst;
            }
        }
    }
    *dstSize = dstSize4 - 3 + tailElems;
}

//---------------------------------------------------------------------------------------------
#define PATTERN1                                                           \
    {                                                                      \
        0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, \
            0xB, -1                                                        \
    }
#define PATTERN2                                                              \
    {                                                                         \
        0x1, 0x0, 0x2, 0x1, 0x4, 0x3, 0x5, 0x4, 0x7, 0x6, 0x8, 0x7, 0xA, 0x9, \
            0xB, 0xA                                                          \
    }

void Base64Encode(const uint8_t* src, size_t size, uint8_t* dst) {
    size_t size3 = AlignLoAny(size, 3);
    size_t size_dst = size3 / 3 * 4;
    if (size > size3) size_dst += 4;

    size_t num_gangs = CeilDiv(size_dst, (size_t)WS);
#psim num_spmd_gangs(num_gangs) gang_size(WS)
    {
        uint32_t lane_id = psim_get_lane_num();
        size_t src_elem_id = 48 * psim_get_gang_num() + lane_id;
        size_t dst_elem_id = psim_get_thread_num();
        uint8_t my_src = 0;
        uint8_t my_dst = 0;

        if (psim_is_tail_gang()) {
            if (src_elem_id < size) {
                my_src = src[src_elem_id];
            }
        } else {
            my_src = src[src_elem_id];
        }

        const int shfl1[] = PATTERN1;
        const int shfl2[] = PATTERN2;

        uint32_t val32 = psim_zip_sync<uint32_t>(my_src);
        uint32_t permute32 =
            psim_shuffle_sync<uint32_t>(val32, shfl1[lane_id % 16]);
        uint8_t permute8 = psim_unzip_sync<uint8_t>(permute32, 0);
        uint8_t shuffle8 = psim_shuffle_sync<uint8_t>(
            permute8, shfl2[lane_id % 16] + lane_id / 16 * 16);
        uint16_t shuffle16 = psim_zip_sync<uint16_t>(shuffle8);
        uint16_t mullo_op = (lane_id % 2 == 0) ? 0x0010 : 0x0100;
        uint16_t mullo = shuffle16 * mullo_op;
        uint16_t mulhi_op = (lane_id % 2 == 0) ? 0x0040 : 0x0400;
        uint16_t mulhi = psim_umulh(shuffle16, mulhi_op);
        uint16_t index16 = (mullo & 0x3f00) | (mulhi & 0x003f);
        uint8_t index = psim_unzip_sync<uint8_t>(index16, 0);

        bool isUpCase = index < 26;
        bool isLetter = index < 52;
        bool isDigit = index < 62;

        uint8_t upValue = isUpCase ? 'A' + index : 0;
        uint8_t lowValue = (isLetter & !isUpCase) ? 'a' + index - 26 : 0;
        uint8_t digValue = (isDigit & !isLetter) ? '0' + index - 52 : 0;
        uint8_t sign_p = index == 62 ? '+' : 0;
        uint8_t sign_d = index == 63 ? '/' : 0;
        my_dst = upValue | lowValue | digValue | sign_p | sign_d;

        if (psim_is_tail_gang()) {
            if (dst_elem_id < size_dst) {
                dst[dst_elem_id] = my_dst;
                if (size - size3) {
                    if (dst_elem_id == size_dst - 1) {
                        dst[dst_elem_id] = '=';
                    }
                }
                if (size - size3 == 1) {
                    if (dst_elem_id == size_dst - 2) {
                        dst[dst_elem_id] = '=';
                    }
                }
            }
        } else {
            dst[dst_elem_id] = my_dst;
        }
    }
}

#endif
}  // namespace Psv
}  // namespace Simd
