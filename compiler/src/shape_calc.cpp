/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "shape_calc.h"
#include "shape.h"

namespace ps {

z3::expr allIndicesZero(Shape s) {
    z3::expr_vector v(s.base.ctx());
    for (auto i : s.indices) {
        v.push_back(i == 0);
    }
    return z3::mk_and(v);
}

z3::expr isMulIndexConstantZero(Shape s, z3::expr base) {
    z3::expr_vector v(s.base.ctx());
    for (auto i : s.indices) {
        v.push_back(i * base == 0);
    }
    return z3::mk_and(v);
}

z3::expr noUnsignedOverflow(Shape s) {
    z3::expr_vector v(s.base.ctx());
    for (auto i : s.indices) {
        v.push_back(z3::bvadd_no_overflow(s.base, i, false));
    }
    return z3::mk_and(v);
}

KnownTransforms::KnownTransforms() {
    binary["add"] = {"add",
                     [](z3::expr a, z3::expr b) { return a + b; },
                     [](unsigned lane, Shape a, Shape b) {
                         return a.indices[lane] + b.indices[lane];
                     },
                     {}};

    binary["and1"] = {
        "and1",
        [](z3::expr a, z3::expr b) { return a & b; },
        [](unsigned i, Shape a, Shape b) { return a.indices[i] & b.base; },
        {[](Shape a, Shape b) { return b.base < 0; },
         [](Shape a, Shape b) { return allIndicesZero(b); },
         [](Shape a, Shape b) { return exprIsPowerOfTwo(-b.base); },
         [](Shape a, Shape b) { return a.base % -b.base == 0; }}};

    binary["and2"] = {
        "and2",
        [](z3::expr a, z3::expr b) { return a & b; },
        [](unsigned i, Shape a, Shape b) { return a.indices[i] & b.base; },
        {[](Shape a, Shape b) { return b.base > 0; },
         [](Shape a, Shape b) { return allIndicesZero(b); },
         [](Shape a, Shape b) { return exprIsPowerOfTwo(b.base + 1); },
         [](Shape a, Shape b) { return a.base % (b.base + 1) == 0; }}};

    binary["and3"] = {"and3",
                      [](z3::expr a, z3::expr b) { return a & b; },
                      [](unsigned i, Shape a, Shape b) {
                          return a.indices[i] & (b.base + b.indices[i]);
                      },
                      {[](Shape a, Shape b) { return allIndicesZero(b); },
                       [](Shape a, Shape b) { return a.base == 0; }}};

    binary["and4"] = {
        "and4",
        [](z3::expr a, z3::expr b) { return a & b; },
        [](unsigned i, Shape a, Shape b) { return a.indices[i] & b.base; },
        {[](Shape a, Shape b) { return b.base > 0; },
         [](Shape a, Shape b) { return allIndicesZero(b); },
         [](Shape a, Shape b) {
             z3::expr_vector v(a.base.ctx());
             for (auto i : a.indices) {
                 v.push_back((a.base & i) == 0);
             }
             return z3::mk_and(v);
         }}};

    binary["ashr"] = {"ashr",
                      [](z3::expr a, z3::expr b) { return z3::ashr(a, b); },
                      [](unsigned i, Shape a, Shape b) {
                          return z3::ashr(a.indices[i], b.base);
                      },
                      {[](Shape a, Shape b) { return b.base > 0; },
                       [](Shape a, Shape b) { return allIndicesZero(b); },
                       {[](Shape a, Shape b) {
                           unsigned width = a.base.get_sort().bv_size();
                           z3::expr max = a.base.ctx().bv_val(
                               (uint64_t)1 << (width - 1), width);
                           return a.base < max;
                       }}}};

    binary["lshr"] = {"lshr",
                      [](z3::expr a, z3::expr b) { return z3::lshr(a, b); },
                      [](unsigned i, Shape a, Shape b) {
                          return z3::lshr(a.indices[i], b.base);
                      },
                      {[](Shape a, Shape b) { return b.base > 0; },
                       [](Shape a, Shape b) { return allIndicesZero(b); },
                       [](Shape a, Shape b) { return noUnsignedOverflow(a); },
                       {[](Shape a, Shape b) {
                           unsigned width = a.base.get_sort().bv_size();
                           z3::expr one = a.base.ctx().bv_val(1, width);
                           return z3::urem(a.base, z3::shl(one, b.base)) == 0;
                       }}}};

    binary["mul1"] = {
        "mul1",
        [](z3::expr a, z3::expr b) { return a * b; },
        [](unsigned i, Shape a, Shape b) { return a.indices[i] * b.base; },
        {[](Shape a, Shape b) {
             return b.base.ctx().bool_val(b.hasConstantBase());
         },
         [](Shape a, Shape b) { return allIndicesZero(b); }}};

    binary["mul2"] = {
        "mul2",
        [](z3::expr a, z3::expr b) { return a * b; },
        [](unsigned i, Shape a, Shape b) { return a.indices[i] * b.base; },
        {[](Shape a, Shape b) { return isMulIndexConstantZero(a, b.base); },
         [](Shape a, Shape b) { return allIndicesZero(b); }}};

    binary["or1"] = {"or1",
                     [](z3::expr a, z3::expr b) { return a | b; },
                     [](unsigned i, Shape a, Shape b) {
                         return (a.indices[i] | b.base) - b.base;
                     },
                     {[](Shape a, Shape b) { return allIndicesZero(b); },
                      [](Shape a, Shape b) {
                          return b.base.ctx().bool_val(b.hasConstantBase());
                      },
                      [](Shape a, Shape b) { return a.base == 0; }}};

    binary["or2"] = {"or2",
                     [](z3::expr a, z3::expr b) { return a | b; },
                     [](unsigned i, Shape a, Shape b) {
                         unsigned width = a.base.get_sort().bv_size();
                         return a.base.ctx().bv_val(0, width);
                     },
                     {[](Shape a, Shape b) { return allIndicesZero(a); },
                      [](Shape a, Shape b) { return allIndicesZero(b); }}};

#if 0
    // FIXME these preconditions aren't sufficient.  If needed, find valid
    // preconditions and use them
    binary["sdiv"] = {
        "sdiv",
        [](z3::expr a, z3::expr b) { return a / b; },
        [](unsigned i, Shape a, Shape b) {
            return z3::udiv(a.indices[i], b.base);
        },
        {[](Shape a, Shape b) { return b.base > 0; },
         [](Shape a, Shape b) { return allIndicesZero(b); },
         [](Shape a, Shape b) { return noUnsignedOverflow(a); },
         [](Shape a, Shape b) { return z3::urem(a.base, b.base) == 0; }}};
#endif

    binary["shl"] = {"shl",
                     [](z3::expr a, z3::expr b) { return z3::shl(a, b); },
                     [](unsigned i, Shape a, Shape b) {
                         return z3::shl(a.indices[i], b.base);
                     },
                     {[](Shape a, Shape b) { return b.base > 0; },
                      [](Shape a, Shape b) { return allIndicesZero(b); }}};

    binary["sub"] = {"sub",
                     [](z3::expr a, z3::expr b) { return a - b; },
                     [](unsigned lane, Shape a, Shape b) {
                         return a.indices[lane] - b.indices[lane];
                     },
                     {}};

    binary["udiv"] = {
        "udiv",
        [](z3::expr a, z3::expr b) { return z3::udiv(a, b); },
        [](unsigned i, Shape a, Shape b) {
            return z3::udiv(a.indices[i], b.base);
        },
        {[](Shape a, Shape b) { return b.base > 0; },
         [](Shape a, Shape b) { return allIndicesZero(b); },
         [](Shape a, Shape b) { return noUnsignedOverflow(a); },
         [](Shape a, Shape b) { return z3::urem(a.base, b.base) == 0; }}};

    binary["urem"] = {
        "urem",
        [](z3::expr a, z3::expr b) { return z3::urem(a, b); },
        [](unsigned i, Shape a, Shape b) {
            return z3::urem(a.indices[i], b.base);
        },
        {[](Shape a, Shape b) { return b.base > 0; },
         [](Shape a, Shape b) { return allIndicesZero(b); },
         [](Shape a, Shape b) { return noUnsignedOverflow(a); },
         [](Shape a, Shape b) { return z3::urem(a.base, b.base) == 0; }}};

    binary["xor"] = {"xor",
                     [](z3::expr a, z3::expr b) { return a ^ b; },
                     [](unsigned i, Shape a, Shape b) {
                         return a.base.ctx().bv_val(
                             0, a.base.get_sort().bv_size());
                     },
                     {[](Shape a, Shape b) { return allIndicesZero(a); },
                      [](Shape a, Shape b) { return allIndicesZero(b); }}};
}

UnaryShapeTransform KnownTransforms::sext(unsigned target_width) {
    return {
        "sext",
        [=](z3::expr a) {
            return z3::sext(a, target_width - a.get_sort().bv_size());
        },
        [=](unsigned i, Shape a) {
            return z3::sext(a.indices[i],
                            target_width - a.base.get_sort().bv_size());
        },
        {[](Shape a) {
            z3::expr_vector v(a.base.ctx());
            z3::expr base = a.base.extract(a.base.get_sort().bv_size() - 2, 0);
            for (auto i : a.indices) {
                z3::expr idx = i.extract(a.base.get_sort().bv_size() - 2, 0);
                v.push_back(z3::bvadd_no_overflow(base, idx, false));
                v.push_back(z3::bvadd_no_overflow(a.base, i, false));
            }
            return z3::mk_and(v);
        }}};
}

UnaryShapeTransform KnownTransforms::trunc(unsigned target_width) {
    return {"trunc",
            [=](z3::expr a) { return a.extract(target_width - 1, 0); },
            [=](unsigned i, Shape a) {
                return a.indices[i].extract(target_width - 1, 0);
            },
            {}};
}

UnaryShapeTransform KnownTransforms::zext(unsigned target_width) {
    return {"zext",
            [=](z3::expr a) {
                return z3::zext(a, target_width - a.get_sort().bv_size());
            },
            [=](unsigned i, Shape a) {
                return z3::zext(a.indices[i],
                                target_width - a.base.get_sort().bv_size());
            },
            {[](Shape a) {
                z3::expr_vector v(a.base.ctx());
                for (auto i : a.indices) {
                    v.push_back(z3::bvadd_no_overflow(a.base, i, false));
                }
                return z3::mk_and(v);
            }}};
}

KnownTransforms known_transforms;

}  // namespace ps
