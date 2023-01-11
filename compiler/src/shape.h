/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#pragma once

#include <z3++.h>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <vector>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Value.h>
#include <llvm/Passes/PassBuilder.h>

#include "utils.h"

namespace ps {

struct MemInstMappedShape {
    enum MappedShape {
        NONE,
        UNIFORM,
        PACKED,
        ALREADY_PACKED,
        PACKED_SHUFFLE,
        GLOBAL_VALUE,
        GATHER_SCATTER
    } mapped_shape;
    uint64_t elem_size;
    std::vector<int> indices;

    MemInstMappedShape() : mapped_shape(NONE), elem_size(0) {}

    std::string toString() {
        std::stringstream s;
        s << "MemInstr: ";
        switch (mapped_shape) {
            case NONE:
                s << "NONE";
                break;
            case UNIFORM:
                s << "UNIFORM";
                break;
            case PACKED:
                s << "PACKED";
                break;
            case ALREADY_PACKED:
                s << "ALREADY_PACKED";
                break;
            case PACKED_SHUFFLE:
                s << "PACKED_SHUFFLE";
                break;
            case GLOBAL_VALUE:
                s << "GLOBAL_VALUE";
                break;
            case GATHER_SCATTER:
                s << "GATHER_SCATTER";
                break;
        }
        s << ", bytes " << std::to_string(elem_size);
        return s.str();
    }
};

class Shape {
  public:
    enum ShapeType { UNKNOWN, NONE, VARYING, INDEXED } type;
    z3::expr base;
    std::vector<z3::expr> indices;
    llvm::GlobalValue* global_value;

    ////////////////////////////////////////////////////////////////////////////
    // Constructors

    Shape(ShapeType type, z3::expr base, std::vector<z3::expr> indices)
        : type(type), base(base), indices(indices), global_value(nullptr) {}

    static Shape Strided(z3::expr base, uint64_t stride, uint32_t num_lanes) {
        Shape s(INDEXED, base, {});
        unsigned width = base.get_sort().bv_size();
        for (uint32_t i = 0; i < num_lanes; i++) {
            s.indices.push_back(constantExpr(base.ctx(), i * stride, width));
        }
        return s;
    }

    static Shape Uniform(z3::expr base, uint32_t num_lanes) {
        return Strided(base, 0, num_lanes);
    }

    Shape(ShapeType type)
        : type(type), base(ctx_for_invalid_bases), global_value(nullptr) {
        ASSERT(type != INDEXED, "Use another constructor for INDEXED");
    }

    static Shape Indexed(z3::expr base, std::vector<z3::expr> indices) {
        return Shape(INDEXED, base, indices);
    }

    static Shape Unknown() { return Shape(UNKNOWN); }
    static Shape Varying() { return Shape(VARYING); }
    static Shape None() { return Shape(NONE); }

    ////////////////////////////////////////////////////////////////////////////
    // z3::expr builders

    static z3::expr constantExpr(z3::context& ctx, uint64_t val,
                                 unsigned width) {
        ASSERT(width <= 64, "Do you really want a type of width >64?");
        return ctx.bv_val(val, width);
    }

    static z3::expr symbolicExpr(z3::context& ctx, std::string name,
                                 unsigned width) {
        ASSERT(width <= 64, "Do you really want a type of width >64?");
        return ctx.bv_const(name.c_str(), width);
    }

    static z3::expr symbolicExpr(z3::solver& solver, std::string name,
                                 unsigned width, uint64_t alignment = 0) {
        ASSERT(width <= 64, "Do you really want a type of width >64?");
        z3::expr e = solver.ctx().bv_const(name.c_str(), width);
        if (alignment > 1) {
            solver.add(z3::urem(e, solver.ctx().bv_val(alignment, width)) == 0);

            /* adding 'mod 3' as a special case, as an optimization hack.  Even
             * though it's redundant with the constraint above, adding this
             * explicitly makes it easier for the solver to check 'mod 3'
             * specifically. */
            if (alignment / 3 * 3 == alignment) {
                solver.add(z3::urem(e, solver.ctx().bv_val(3, width)) == 0);
            }
        }
        return e;
    }

    ////////////////////////////////////////////////////////////////////////////

    bool isUniform() const {
        if (type != INDEXED) {
            return false;
        }
        if (isStrided() && getStride() == 0) {
            return true;
        }
        return false;
    }

    bool isIndexed() const { return type == INDEXED; }
    bool isVarying() const { return type == VARYING || type == UNKNOWN; }
    bool isUnknown() const { return type == UNKNOWN; }
    bool isNone() const { return type == NONE; }

    bool hasConstantBase() const {
        if (!isIndexed()) {
            return false;
        }
        uint64_t val;
        bool success = base.simplify().is_numeral_u64(val);
        return success;
    }

    uint64_t getConstantBase() const {
        uint64_t val;
        bool success = base.simplify().is_numeral_u64(val);
        ASSERT(success,
               "Base is not constant: " << base.simplify().to_string());
        return val;
    }

    z3::expr getExprAtLane(unsigned i) const { return base + indices[i]; }

    uint64_t getValueAtLane(unsigned i) const {
        uint64_t val;
        bool success = (base + indices[i]).simplify().is_numeral_u64(val);
        assert(success);
        return val;
    }

    uint64_t getIndexAsInt(unsigned i) const {
        uint64_t val;
        bool success = indices[i].simplify().is_numeral_u64(val);
        assert(success);
        return val;
    }

    std::vector<uint64_t> getIndicesAsInts() const {
        std::vector<uint64_t> v;
        for (unsigned i = 0; i < indices.size(); i++) {
            v.push_back(getIndexAsInt(i));
        }
        return v;
    }

    Shape eval(z3::model m) {
        if (type != INDEXED) {
            return *this;
        }

        std::vector<z3::expr> v;
        for (auto i : indices) {
            v.push_back(m.eval(i, true));
        }
        return Indexed(m.eval(base), v);
    }

    bool isStrided() const {
        uint64_t stride;
        return getStride(stride);
    }

    uint64_t getStride() const {
        uint64_t stride;
        bool ret = getStride(stride);
        if (!ret) {
            FATAL("Shape " << toString() << " is not strided\n");
        }
        return stride;
    }

    bool isGangPacked(size_t elem_size);
    int64_t getMaxIndex();
    int64_t getMinIndex();

    bool operator==(const Shape& other) const {
        if (type != other.type) {
            return false;
        }

        z3::expr_vector v(base.ctx());
        for (unsigned i = 0; i < indices.size(); i++) {
            v.push_back(indices[i] == other.indices[i]);
        }
        return (base == other.base).simplify().is_true() &&
               z3::mk_and(v).simplify().is_true();
    }
    bool operator!=(const Shape& other) const { return !(*this == other); }

    std::string toString(bool symbolic_indices = false) const;

  protected:
    bool getStride(uint64_t& stride) const {
        if (type != INDEXED) {
            return false;
        }

        ASSERT(!indices.empty(), "getStride on Shape with no indices");
        if (indices.size() == 1) {
            stride = 0;
            return true;
        }

        std::vector<uint64_t> v = getIndicesAsInts();

        stride = v[1] - v[0];
        for (size_t i = 1; i < v.size(); i++) {
            if (stride != v[i] - v[i - 1]) {
                return false;
            }
        }
        return true;
    }

    static z3::context ctx_for_invalid_bases;
};

}  // namespace ps
