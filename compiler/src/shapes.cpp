/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "shapes.h"

#include <llvm/IR/Function.h>
#include <llvm/Passes/PassBuilder.h>

#include <chrono>
#include <iomanip>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "shape.h"
#include "utils.h"
#include "vectorize.h"

using namespace llvm;

namespace ps {

unsigned shapes_verbosity_level;
[[maybe_unused]] static unsigned& verbosity_level = shapes_verbosity_level;

////////////////////////////////////////////////////////////////////////////////

ShapesStep::ShapesStep(VectorizedFunctionInfo& vf_info)
    : vf_info(vf_info),
      value_cache(vf_info.value_cache),
      num_lanes(vf_info.vfabi.vlen) {}

template <typename T, typename... S>
Shape ShapesStep::tryTransform(std::vector<T> transforms, Shape sa,
                               S... other_shapes) {
    z3::solver& s = vf_info.solver;

    for (auto& t : transforms) {
        PRINT_HIGH("Checking shape transform " << t.name);

        bool assumptions_confirmed = true;
        for (auto f : t.assumptions) {
            z3::expr assumption = f(sa, other_shapes...).simplify();
            PRINT_HIGH("Checking assumption " << assumption.to_string());
            if (assumption.is_true()) {
                PRINT_HIGH(
                    "Assumption can be proven via simplification; "
                    "don't even need to run the solver");
                continue;
            }
            if (assumption.is_false()) {
                PRINT_HIGH(
                    "Assumption can be disproven via simplification; "
                    "don't even need to run the solver");
                assumptions_confirmed = false;
                break;
            }

            for (auto& i : {sa, other_shapes...}) {
                PRINT_HIGH(i.toString());
            }
            DEBUG_HIGH(llvm::errs().flush());

            s.push();
            s.add(!assumption);

            auto t_before = std::chrono::high_resolution_clock::now();
            z3::check_result r = s.check();
            auto t_after = std::chrono::high_resolution_clock::now();
            auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                t_after - t_before);
            if (verbosity_level >= 3 || t_diff.count() > 1000) {
                PRINT_ALWAYS("Shape transform '" << t.name
                                                 << "' assumption check took "
                                                 << t_diff.count() << "ms");
                PRINT_HIGH("Solver had " << vf_info.solver.assertions().size()
                                         << " assertions");
                for (auto i : vf_info.solver.assertions()) {
                    PRINT_HIGH("  " << i.simplify().to_string());
                }
            }

            switch (r) {
                case z3::sat:
                    PRINT_HIGH(
                        "Found counterexample to assumption for shape "
                        "transform "
                        << t.name);
                    if (verbosity_level >= 3) {
                        z3::model m = s.get_model();
                        for (Shape i : {sa, other_shapes...}) {
                            PRINT_HIGH(i.toString());
                            PRINT_HIGH(" = " << i.eval(m).toString());
                        }
                    }
                    assumptions_confirmed = false;
                    break;

                case z3::unknown:
                    PRINT_HIGH(
                        "Solver returned unknown checking assumption for shape "
                        "transform "
                        << t.name);
                    assumptions_confirmed = false;
                    break;

                case z3::unsat:
                    break;

                default:
                    ASSERT(false, "unreachable");
            }

            s.pop();
            if (!assumptions_confirmed) {
                break;
            }
        }

        if (assumptions_confirmed) {
            PRINT_HIGH("Shape transform " << t.name << " legality confirmed");
            std::vector<z3::expr> indices;
            for (unsigned i = 0; i < sa.indices.size(); i++) {
                z3::expr index_expr =
                    t.f_proposed_index(i, sa, other_shapes...).simplify();
                uint64_t as_uint64;
                ASSERT(index_expr.is_numeral_u64(as_uint64),
                       "Could not simplify index " << index_expr.to_string()
                                                   << " to a numeric value");
                indices.push_back(index_expr);
            }
            return Shape::Indexed(t.f_expr(sa.base, other_shapes.base...),
                                  indices);
        }
    }

    PRINT_HIGH("No valid transform found");
    return Shape::Varying();
}

Shape ShapesStep::transformKnownBases(std::function<z3::expr(z3::expr)> f,
                                      Shape sa) {
    PRINT_HIGH("Transforming shape with known bases");
    std::vector<z3::expr> indices;
    for (unsigned i = 0; i < sa.indices.size(); i++) {
        z3::expr index_expr =
            (f(sa.base + sa.indices[i]) - f(sa.base)).simplify();
        uint64_t as_uint64;
        ASSERT(index_expr.is_numeral_u64(as_uint64),
               "Could not simplify index " << index_expr.to_string()
                                           << " to a numeric value");
        indices.push_back(index_expr);
    }
    PRINT_HIGH("transformed base is " << f(sa.base).simplify().to_string());
    return Shape::Indexed(f(sa.base), indices);
}

Shape ShapesStep::transformKnownBases(
    std::function<z3::expr(z3::expr, z3::expr)> f, Shape sa, Shape sb) {
    PRINT_HIGH("Transforming shape with known bases");

    z3::expr base = f(sa.base, sb.base);
    if (base.is_bool()) {
        base = z3::ite(base, sa.base.ctx().bv_val(1, 1),
                       sa.base.ctx().bv_val(0, 1));
    }

    std::vector<z3::expr> indices;
    for (unsigned i = 0; i < sa.indices.size(); i++) {
        z3::expr actual = f(sa.base + sa.indices[i], sb.base + sb.indices[i]);
        if (actual.is_bool()) {
            actual = z3::ite(actual, sa.base.ctx().bv_val(1, 1),
                             sa.base.ctx().bv_val(0, 1));
        }
        z3::expr index_expr = (actual - base).simplify();
        uint64_t as_uint64;
        ASSERT(index_expr.is_numeral_u64(as_uint64),
               "Could not simplify index " << index_expr.to_string()
                                           << " to a numeric value");
        indices.push_back(index_expr);
    }
    return Shape::Indexed(base, indices);
}

Shape ShapesStep::transformKnownBases(
    std::function<z3::expr(z3::expr, z3::expr, z3::expr)> f, Shape sa, Shape sb,
    Shape sc) {
    PRINT_HIGH("Transforming shape with known bases");

    z3::expr base = f(sa.base, sb.base, sc.base);

    std::vector<z3::expr> indices;
    for (unsigned i = 0; i < sa.indices.size(); i++) {
        z3::expr actual = f(sa.base + sa.indices[i], sb.base + sb.indices[i],
                            sc.base + sc.indices[i]);
        z3::expr index_expr = (actual - base).simplify();
        uint64_t as_uint64;
        ASSERT(index_expr.is_numeral_u64(as_uint64),
               "Could not simplify index " << index_expr.to_string()
                                           << " to a numeric value");
        indices.push_back(index_expr);
    }
    return Shape::Indexed(base, indices);
}

Shape ShapesStep::calculateShapeBinaryOp(BinaryOperator* binop) {
    Value* a = binop->getOperand(0);
    Value* b = binop->getOperand(1);

    Shape sa = value_cache.getShape(a);
    Shape sb = value_cache.getShape(b);

    /* early checks common to all BinaryOps*/
    if (sa.isUnknown() || sb.isUnknown()) {
        return Shape::Unknown();
    } else if (sa.isVarying() || sb.isVarying()) {
        return Shape::Varying();
    }
    assert(sb.isIndexed() && sa.isIndexed());
    assert(sa.indices.size() == sb.indices.size());

    std::vector<uint64_t> indices;

    /* If we know what the base values are at compile time, just do the math */
    if (sa.hasConstantBase() && sb.hasConstantBase()) {
        switch (binop->getOpcode()) {
            case BinaryOperator::Add:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return a + b; }, sa, sb);
            case BinaryOperator::And:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return a & b; }, sa, sb);
            case BinaryOperator::AShr:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return z3::ashr(a, b); }, sa,
                    sb);
            case BinaryOperator::LShr:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return z3::lshr(a, b); }, sa,
                    sb);
            case BinaryOperator::Mul:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return a * b; }, sa, sb);
            case BinaryOperator::Or:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return a | b; }, sa, sb);
            case BinaryOperator::SDiv:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return a / b; }, sa, sb);
            case BinaryOperator::SRem:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return z3::srem(a, b); }, sa,
                    sb);
            case BinaryOperator::Sub:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return a - b; }, sa, sb);
            case BinaryOperator::UDiv:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return z3::udiv(a, b); }, sa,
                    sb);
            case BinaryOperator::URem:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return z3::urem(a, b); }, sa,
                    sb);
            case BinaryOperator::Xor:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return a ^ b; }, sa, sb);
            default:
                break;
        }
    }

    /* Use information about no-[un]signed-wrap where possible */
    for (unsigned i = 0; i < sa.indices.size(); i++) {
        if (binop->getOpcode() == BinaryOperator::Add) {
            if (binop->hasNoSignedWrap()) {
                vf_info.solver.add(z3::bvadd_no_overflow(
                    sa.getExprAtLane(i), sb.getExprAtLane(i), true));
                vf_info.solver.add(z3::bvadd_no_underflow(sa.getExprAtLane(i),
                                                          sb.getExprAtLane(i)));
            }
            if (binop->hasNoUnsignedWrap()) {
                vf_info.solver.add(z3::bvadd_no_overflow(
                    sa.getExprAtLane(i), sb.getExprAtLane(i), false));
            }
        } else if (binop->getOpcode() == BinaryOperator::Sub) {
            if (binop->hasNoSignedWrap()) {
                vf_info.solver.add(z3::bvsub_no_overflow(sa.getExprAtLane(i),
                                                         sb.getExprAtLane(i)));
                vf_info.solver.add(z3::bvsub_no_underflow(
                    sa.getExprAtLane(i), sb.getExprAtLane(i), true));
            }
            if (binop->hasNoUnsignedWrap()) {
                vf_info.solver.add(z3::bvsub_no_underflow(
                    sa.getExprAtLane(i), sb.getExprAtLane(i), false));
            }
        } else if (binop->getOpcode() == BinaryOperator::Mul) {
            if (binop->hasNoSignedWrap()) {
                vf_info.solver.add(z3::bvmul_no_overflow(
                    sa.getExprAtLane(i), sb.getExprAtLane(i), true));
                vf_info.solver.add(z3::bvmul_no_underflow(sa.getExprAtLane(i),
                                                          sb.getExprAtLane(i)));
            }
            if (binop->hasNoUnsignedWrap()) {
                vf_info.solver.add(z3::bvmul_no_overflow(
                    sa.getExprAtLane(i), sb.getExprAtLane(i), false));
            }
        }
    }

    /* Try the transforms proven offline using z3 */
    switch (binop->getOpcode()) {
        case BinaryOperator::Add:
            return tryTransform<BinaryShapeTransform>(
                {known_transforms.binary["add"]}, sa, sb);

        case BinaryOperator::And:
            return tryTransform<BinaryShapeTransform>(
                {known_transforms.binary["and1"],
                 known_transforms.binary["and2"],
                 known_transforms.binary["and3"],
                 known_transforms.binary["and4"]},
                sa, sb);

        case BinaryOperator::AShr:
            return tryTransform<BinaryShapeTransform>(
                {known_transforms.binary["ashr"]}, sa, sb);

        case BinaryOperator::FAdd:
        case BinaryOperator::FSub:
            return Shape::Varying();

        case BinaryOperator::Mul:
            return tryTransform<BinaryShapeTransform>(
                {known_transforms.binary["mul1"],
                 known_transforms.binary["mul2"]},
                sa, sb);

        case BinaryOperator::Or:
            return tryTransform<BinaryShapeTransform>(
                {known_transforms.binary["or1"],
                 known_transforms.binary["or2"]},
                sa, sb);

        case BinaryOperator::LShr:
            return tryTransform<BinaryShapeTransform>(
                {known_transforms.binary["lshr"]}, sa, sb);

#if 0
        case BinaryOperator::SDiv:
            return tryTransform<BinaryShapeTransform>(
                {known_transforms.binary["sdiv"]}, sa, sb);
#endif

        case BinaryOperator::Shl:
            return tryTransform<BinaryShapeTransform>(
                {known_transforms.binary["shl"]}, sa, sb);

        case BinaryOperator::Sub:
            return tryTransform<BinaryShapeTransform>(
                {known_transforms.binary["sub"]}, sa, sb);

        case BinaryOperator::UDiv:
            return tryTransform<BinaryShapeTransform>(
                {known_transforms.binary["udiv"]}, sa, sb);

        case BinaryOperator::URem:
            return tryTransform<BinaryShapeTransform>(
                {known_transforms.binary["urem"]}, sa, sb);

        case BinaryOperator::Xor:
            return tryTransform<BinaryShapeTransform>(
                {known_transforms.binary["xor"]}, sa, sb);

        default:
            return Shape::Unknown();
    }
}

Shape ShapesStep::calculateShapeCall(CallInst* call) {
    Function* f = call->getCalledFunction();

    if (!f || !f->hasName()) {
        // Function pointers return nullptr here because they don't point to any
        // particular function at compile time.  In this case, just return shape
        // Unknown
        return Shape::Unknown();
    }

    FunctionResolver::PsimApiEnum psim_api_enum =
        vf_info.vm_info.function_resolver.getPsimApiEnum(f);
    switch (psim_api_enum) {
        case FunctionResolver::PsimApiEnum::GET_LANE_NUM:
            return Shape::Strided(Shape::constantExpr(vf_info.z3_ctx, 0, 32), 1,
                                  num_lanes);
        case FunctionResolver::PsimApiEnum::GET_THREAD_NUM: {
            z3::expr thread_num = Shape::symbolicExpr(
                vf_info.solver, "thread_num", 64, num_lanes);
            vf_info.solver.add(z3::ult(thread_num, INT64_MAX - num_lanes));
            vf_info.solver.add(z3::sge(thread_num, 0));
            return Shape::Strided(thread_num, 1, num_lanes);
        } break;
        case FunctionResolver::PsimApiEnum::GET_GANG_SIZE:
            return Shape::Uniform(
                Shape::constantExpr(vf_info.z3_ctx, num_lanes, 32), num_lanes);
        case FunctionResolver::PsimApiEnum::GET_GANG_NUM:
            return Shape::Uniform(
                Shape::symbolicExpr(vf_info.z3_ctx, "gang_num", 64), num_lanes);
        case FunctionResolver::PsimApiEnum::GET_GRID_SIZE:
            return Shape::Uniform(
                Shape::symbolicExpr(vf_info.z3_ctx, "grid_size", 64),
                num_lanes);
        case FunctionResolver::PsimApiEnum::GET_OMP_THREAD_NUM:
            return Shape::Uniform(
                Shape::symbolicExpr(vf_info.z3_ctx, "omp_thread_num", 32),
                num_lanes);
        case FunctionResolver::PsimApiEnum::SADD_SAT:
        case FunctionResolver::PsimApiEnum::UADD_SAT:
        case FunctionResolver::PsimApiEnum::SSUB_SAT:
        case FunctionResolver::PsimApiEnum::USUB_SAT:
            for (Use& arg : call->args()) {
                if (arg->getType()->isMetadataTy()) {
                    continue;
                }
                if (!value_cache.getShape(arg).isUniform()) {
                    return Shape::Varying();
                }
            }
            return Shape::Uniform(
                Shape::symbolicExpr(vf_info.z3_ctx, call->getName().str(),
                                    getValueSizeBits(call)),
                num_lanes);
        // force UMULH to be varying for now
        case FunctionResolver::PsimApiEnum::UMULH:
        // *_SYNC is always varying since it is a collective
        case FunctionResolver::PsimApiEnum::SHFL_SYNC:
        case FunctionResolver::PsimApiEnum::ZIP_SYNC:
        case FunctionResolver::PsimApiEnum::UNZIP_SYNC:
            return Shape::Varying();
        // force ATOMICADD_LOCAL to be none for now
        case FunctionResolver::PsimApiEnum::ATOMICADD_LOCAL:
        case FunctionResolver::PsimApiEnum::GANG_SYNC:
        case FunctionResolver::PsimApiEnum::COLLECTIVE_ADD_ABS_DIFF:
            return Shape::None();
        case FunctionResolver::PsimApiEnum::PSIM_API_NONE:
            break;
    }
    return Shape::Varying();
}

Shape ShapesStep::calculateShapeGEP(GetElementPtrInst* gep) {
    Shape shape = value_cache.getShape(gep->getPointerOperand());
    Type* ty = gep->getPointerOperand()->getType();
    PRINT_HIGH("GEP source element type is " << *ty);
    PRINT_HIGH("GEP pointer has shape " << shape.toString());

    if (shape.isUnknown()) {
        return Shape::Unknown();
    }

    if (shape.isVarying()) {
        return Shape::Varying();
    }

    for (Value* v : gep->indices()) {
        // Check the shape of the index
        Shape sv = Shape::Unknown();
        if (isa<Instruction>(v) && !value_cache.has(v)) {
            PRINT_HIGH("Operand " << *v << " is not yet available; "
                                  << "assuming Uniform for now");
            sv = Shape::Uniform(
                Shape::symbolicExpr(vf_info.z3_ctx, v->getName().str(),
                                    getValueSizeBits(v)),
                num_lanes);
        } else {
            sv = value_cache.getShape(v);
        }
        PRINT_HIGH("Index " << *v << " has shape " << sv.toString());
        if (sv.isUnknown()) {
            return Shape::Unknown();
        }

        if (sv.isVarying()) {
            PRINT_HIGH("Index " << *v << " is varying");
            return Shape::Varying();
        }

        assert(sv.indices.size() == shape.indices.size());

        // Dereference one level deeper into the GEP indexing
        PointerType* p_ty = dyn_cast<PointerType>(ty);
        StructType* s_ty = dyn_cast<StructType>(ty);
        if (p_ty) {
            ty = p_ty->getNonOpaquePointerElementType();
        } else if (s_ty) {
            if (!sv.isUniform() || !sv.hasConstantBase()) {
                PRINT_HIGH(
                    "Index determining struct element is not uniform with "
                    "known base; returning Varying");
                return Shape::Varying();
            }

            // since the shape is uniform, it's safe to just use base here
            ty = GetElementPtrInst::getTypeAtIndex(ty, sv.getConstantBase());
        } else {
            // for vector or array types, it doesn't matter which position in
            // the array we dereference to figure out the type
            ty = GetElementPtrInst::getTypeAtIndex(ty, (uint64_t)0);
        }

        ASSERT(ty, "Could not determine next indexed type");
        PRINT_HIGH("Indexed type is " << *ty);

        // Calculate the size of the type at the current level of dereferencing
        int64_t s = vf_info.data_layout.getTypeAllocSize(ty);
        z3::expr b = Shape::constantExpr(shape.base.ctx(), s,
                                         shape.base.get_sort().bv_size());
        PRINT_HIGH("Indexed type " << *ty << " has layout size " << s << " "
                                   << b.simplify().to_string());

        // Do the arithmetic to update the shape
        for (unsigned i = 0; i < sv.indices.size(); i++) {
            z3::expr idx = sv.indices[i];
            if (b.get_sort().bv_size() > idx.get_sort().bv_size()) {
                idx = z3::zext(
                    idx, b.get_sort().bv_size() - idx.get_sort().bv_size());
            }
            shape.indices[i] = shape.indices[i] + idx * b;
        }
        z3::expr base = sv.base;
        if (b.get_sort().bv_size() > base.get_sort().bv_size()) {
            base = z3::zext(base,
                            b.get_sort().bv_size() - base.get_sort().bv_size());
        }
        shape.base = shape.base + base * b;
        PRINT_HIGH("New shape is " << shape.toString());
    }
    return shape;
}

/* Checks if array layout optimization could be applied for allocas */
bool ShapesStep::analyzeUses(Instruction* inst) {
    for (User* U : inst->users()) {
        if (isa<LoadInst>(U))
            continue;
        else if (isa<BitCastInst>(U)) {
            /* Recursively check users of bcast */
            if (analyzeUses(cast<Instruction>(U))) continue;
            return false;
        } else if (isa<GetElementPtrInst>(U)) {
            /* Disable opt for alloca->gep->gep->... for now. */
            if (isa<GetElementPtrInst>(inst)) return false;
            /* Recursively check users of gep */
            if (analyzeUses(cast<Instruction>(U))) continue;
            return false;
        } else if (CallInst* call = dyn_cast<CallInst>(U)) {
            /* Conservatively return false for
                all CallInsts except intrinsics */
            if (call->getCalledFunction()->isIntrinsic()) continue;
            return false;
        } else if (StoreInst* store = dyn_cast<StoreInst>(U)) {
            // pointers cannot escape
            if (store->getValueOperand() != inst) continue;
            return false;
        } else {
            // Conservatively return false for other insts
            return false;
        }
    }
    return true;
}

/* Generates array layout optimized allocas and geps. */
llvm::Instruction* ShapesStep::generateOptInsts(
    AllocaInst* inst,
    std::set<std::pair<Instruction*, Instruction*>>& toReplace) {
    ArrayType* arr_ty = dyn_cast<ArrayType>(inst->getAllocatedType());
    Type* new_array_type = ArrayType::get(arr_ty->getElementType(), num_lanes);
    new_array_type = ArrayType::get(new_array_type, arr_ty->getNumElements());
    Instruction* new_alloca =
        new AllocaInst(new_array_type, 0, inst->getArraySize(),
                       inst->getAlign(), inst->getName() + ".");
    PRINT_HIGH("Array layout Opt -- New alloca is: " << *new_alloca);
    toReplace.insert(std::make_pair(inst, new_alloca));

    for (User* U : inst->users()) {
        GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(U);
        if (!gep) continue;
        std::vector<Value*> idxlist;
        for (Value* v : gep->indices()) {
            idxlist.push_back(v);
        }
        Function* lane_num = vf_info.mod->getFunction("psim_get_lane_num");
        Instruction* get_lane_id =
            CallInst::Create(lane_num, gep->getName() + ".");
        idxlist.push_back(get_lane_id);
        Instruction* new_gep = GetElementPtrInst::Create(
            new_array_type, new_alloca, idxlist, gep->getName() + ".");
        toReplace.insert(std::make_pair(gep, new_gep));
    }
    return new_alloca;
}

/* Replaces allocas and geps with array layout optimized versions. */
void ShapesStep::insertOptInsts(
    std::set<std::pair<Instruction*, Instruction*>>& toReplace) {
    for (auto& i : toReplace) {
        PRINT_HIGH("Array layout Opt -- Replacing: " << *i.first
                                                     << " With: " << *i.second);
        if (GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(i.second)) {
            // We know that the last operand of the new gep should be getlanenum
            // call
            Value* val = gep->getOperand(gep->getNumOperands() - 1);
            CallInst* call = dyn_cast<CallInst>(val);
            if (!call || vf_info.vm_info.function_resolver.getPsimApiEnum(
                             call->getCalledFunction()) !=
                             FunctionResolver::PsimApiEnum::GET_LANE_NUM)
                FATAL(
                    "Array layout opt -- Last index of gep not a getlanenum "
                    "call"
                    << *val);

            call->insertBefore(i.first);
            vf_info.instruction_order.insert(
                std::find(std::begin(vf_info.instruction_order),
                          std::end(vf_info.instruction_order), i.first),
                call);
        }
        i.second->insertAfter(i.first);
        i.first->replaceAllUsesWith(i.second);
        std::replace(vf_info.instruction_order.begin(),
                     vf_info.instruction_order.end(), i.first, i.second);
        i.first->eraseFromParent();
    }
}

/* Checks for and applies array layout optimization for allocas. */
void ShapesStep::arrayLayoutOpt() {
    std::set<std::pair<Instruction*, Instruction*>> toReplace;
    for (Instruction* I : vf_info.instruction_order) {
        AllocaInst* alloca = dyn_cast<AllocaInst>(I);
        if (!alloca) continue;
        ArrayType* ty = dyn_cast<ArrayType>(alloca->getAllocatedType());
        if (!ty) continue;
        if (ty->getElementType()->isStructTy()) continue;
        if (!analyzeUses(I)) continue;
        PRINT_HIGH("Array layout Opt -- Optimizing alloca " << *alloca);
        Instruction* new_alloca = generateOptInsts(alloca, toReplace);
        z3::expr base =
            Shape::symbolicExpr(vf_info.z3_ctx, new_alloca->getName().str(),
                                getValueSizeBits(new_alloca));
        value_cache.setShape(new_alloca, Shape::Uniform(base, num_lanes));
        value_cache.setArrayLayoutOpt(new_alloca);
    }

    insertOptInsts(toReplace);
}

Shape ShapesStep::calculateShapeCmp(ICmpInst* cmp) {
    Value* a = cmp->getOperand(0);
    Value* b = cmp->getOperand(1);

    Shape sa = value_cache.getShape(a);
    Shape sb = value_cache.getShape(b);

    if (sa.isUnknown() || sb.isUnknown()) {
        return Shape::Unknown();
    }

    if (sa.isVarying() || sb.isVarying()) {
        return Shape::Varying();
    }

    CmpInst::Predicate pred = cmp->getPredicate();

    if (sa.hasConstantBase() && sb.hasConstantBase()) {
        switch (pred) {
            case CmpInst::Predicate::ICMP_NE:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return a != b; }, sa, sb);
            case CmpInst::Predicate::ICMP_EQ:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return a == b; }, sa, sb);
            case CmpInst::Predicate::ICMP_UGT:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return z3::ugt(a, b); }, sa,
                    sb);
            case CmpInst::Predicate::ICMP_ULT:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return z3::ult(a, b); }, sa,
                    sb);
            case CmpInst::Predicate::ICMP_SLT:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return a < b; }, sa, sb);
            case CmpInst::Predicate::ICMP_SLE:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return a <= b; }, sa, sb);
            case CmpInst::Predicate::ICMP_SGT:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return a > b; }, sa, sb);
            case CmpInst::Predicate::ICMP_SGE:
                return transformKnownBases(
                    [](z3::expr a, z3::expr b) { return a >= b; }, sa, sb);
            default:
                WARNING(getDebugLocStr(cmp) +
                            " Don't know how to calculate shape for "
                        << *cmp << " with known operands");
                return Shape::Varying();
        }
    }

    if (sa.isUniform() && sb.isUniform()) {
        switch (pred) {
            case CmpInst::Predicate::ICMP_EQ:
                return Shape::Uniform(
                    z3::ite(sa.base == sb.base, sa.base.ctx().bv_val(1, 1),
                            sa.base.ctx().bv_val(0, 1)),
                    num_lanes);
            case CmpInst::Predicate::ICMP_NE:
                return Shape::Uniform(
                    z3::ite(sa.base != sb.base, sa.base.ctx().bv_val(1, 1),
                            sa.base.ctx().bv_val(0, 1)),
                    num_lanes);
            case CmpInst::Predicate::ICMP_UGT:
                return Shape::Uniform(z3::ite(z3::ugt(sa.base, sb.base),
                                              sa.base.ctx().bv_val(1, 1),
                                              sa.base.ctx().bv_val(0, 1)),
                                      num_lanes);
            case CmpInst::Predicate::ICMP_ULT:
                return Shape::Uniform(z3::ite(z3::ult(sa.base, sb.base),
                                              sa.base.ctx().bv_val(1, 1),
                                              sa.base.ctx().bv_val(0, 1)),
                                      num_lanes);
            case CmpInst::Predicate::ICMP_SLT:
                return Shape::Uniform(
                    z3::ite(sa.base < sb.base, sa.base.ctx().bv_val(1, 1),
                            sa.base.ctx().bv_val(0, 1)),
                    num_lanes);
            case CmpInst::Predicate::ICMP_SLE:
                return Shape::Uniform(
                    z3::ite(sa.base <= sb.base, sa.base.ctx().bv_val(1, 1),
                            sa.base.ctx().bv_val(0, 1)),
                    num_lanes);
            case CmpInst::Predicate::ICMP_SGT:
                return Shape::Uniform(
                    z3::ite(sa.base > sb.base, sa.base.ctx().bv_val(1, 1),
                            sa.base.ctx().bv_val(0, 1)),
                    num_lanes);
            case CmpInst::Predicate::ICMP_SGE:
                return Shape::Uniform(
                    z3::ite(sa.base >= sb.base, sa.base.ctx().bv_val(1, 1),
                            sa.base.ctx().bv_val(0, 1)),
                    num_lanes);
            default:
                WARNING(getDebugLocStr(cmp) +
                            " Don't know how to calculate shape for "
                        << *cmp << " with known operands");
                return Shape::Varying();
        }
    }

    return Shape::Varying();
}

Shape ShapesStep::calculateShapeLoad(LoadInst* load) {
    Shape shape = value_cache.getShape(load->getPointerOperand());
    PRINT_HIGH("Pointer operand has shape " << shape.toString());
    if (shape.isUniform() && !load->getType()->isVectorTy()) {
        return Shape::Uniform(
            Shape::symbolicExpr(vf_info.z3_ctx, load->getName().str(),
                                getValueSizeBits(load)),
            num_lanes);
    }

    if (!shape.isIndexed()) {
        return Shape::Varying();
    }

    GlobalValuePlusOffset gv_plus_offset =
        getGlobalValuePlusOffsetFromExpr(shape.base);
    if (gv_plus_offset.gv) {
        PRINT_HIGH("Load base is global value " << *gv_plus_offset.gv);

        std::vector<uint64_t> values =
            getValuesFromGlobalConstant(gv_plus_offset.gv);

        unsigned width = getBaseValueSizeBytes(gv_plus_offset.gv);
        PRINT_HIGH("Element width is " << width);

        std::vector<z3::expr> indices;
        for (unsigned i = 0; i < shape.indices.size(); i++) {
            unsigned offset_bytes =
                gv_plus_offset.offset + shape.getIndexAsInt(i);
            unsigned idx = offset_bytes / width;
            ASSERT(idx * width == offset_bytes,
                   "Index " << idx << " is not a multiple of the type width "
                            << width);
            ASSERT(idx >= 0 && idx < values.size(),
                   "Index is out of bounds: 0 <= " << idx << " < "
                                                   << values.size());
            indices.push_back(Shape::constantExpr(shape.base.ctx(), values[idx],
                                                  getValueSizeBits(load)));
        }
        z3::expr base =
            Shape::constantExpr(shape.base.ctx(), 0, getValueSizeBits(load));
        Shape s = Shape::Indexed(base, indices);
        s.global_value = gv_plus_offset.gv;
        return s;
    }

    return Shape::Varying();
}

Shape ShapesStep::calculateShapePHI(PHINode* phi) {
    if (phi->getNumIncomingValues() == 1) {
        return value_cache.getShape(phi->getOperand(0));
    }

    /* Step 0: if one of the operand shapes is not yet available,
     * tentatively just set this to uniform.  When the shape of the operand
     * is calculated later, it will re-queue this PHI in the work queue and
     * recalculate its shape properly.
     */
    for (Value* v : phi->operands()) {
        if (isa<Instruction>(v) && !value_cache.has(v)) {
            PRINT_HIGH("Operand " << *v << " is not yet available; "
                                  << "assuming Uniform for now");
            return Shape::Uniform(
                Shape::symbolicExpr(vf_info.z3_ctx, v->getName().str(),
                                    getValueSizeBits(v)),
                num_lanes);
        }
    }

    /* Step 1: if both input values have the same shape, then use that shape
     * as the input to step 2.  Otherwise, the shape is varying.
     */
    assert(phi->getNumIncomingValues() == 2);
    bool is_inverted = false;
    Value* condition = vf_info.getPHISelectMask(phi, &is_inverted);
    assert(condition);

    if (isa<Instruction>(condition) && !value_cache.has(condition)) {
        PRINT_HIGH("Condition " << *condition << " is not yet available; "
                                << "assuming Uniform for now");
        return Shape::Uniform(
            Shape::symbolicExpr(vf_info.z3_ctx, condition->getName().str(),
                                getValueSizeBits(condition)),
            num_lanes);
    }

    Value* a = phi->getOperand(0);
    Value* b = phi->getOperand(1);
    assert(phi->getNumIncomingValues() == 2);
    Shape sa = value_cache.getShape(a);
    Shape sb = value_cache.getShape(b);
    Shape sc = value_cache.getShape(condition);

    if (isa<UndefValue>(a)) {
        return sb;
    } else if (isa<UndefValue>(b)) {
        return sa;
    }

    PRINT_HIGH("pulled shape a " << sa.toString());
    PRINT_HIGH("pulled shape b " << sb.toString());
    PRINT_HIGH("pulled shape c " << sc.toString());
    if (sc.hasConstantBase() && sa.hasConstantBase() && sb.hasConstantBase()) {
        if (is_inverted) {
            return transformKnownBases(
                [](z3::expr c, z3::expr a, z3::expr b) {
                    return z3::ite(c == 1, a, b);
                },
                sc, sa, sb);
        } else {
            return transformKnownBases(
                [](z3::expr c, z3::expr a, z3::expr b) {
                    return z3::ite(c == 1, b, a);
                },
                sc, sa, sb);
        }
    }

    if (!sa.isIndexed() || !sb.isIndexed()) {
        PRINT_HIGH("a and b shapes mismatch, setting shape Varying()");
        return Shape::Varying();
    }
    for (unsigned i = 0; i < sa.indices.size(); i++) {
        if (sa.getIndexAsInt(i) != sb.getIndexAsInt(i)) {
            PRINT_HIGH("a and b indices mismatch, setting shape Varying()");
            return Shape::Varying();
        }
    }

    /* Step 2: if the shape is not already varying, check whether diverging
     * control flow reconverge forces it to become varying:
     *
     * BB0:
     * A = ...
     * if (cond) goto BB1 else BB2
     *  |     \
     *  |     BB1:
     *  |     B = ...
     *  |     /
     *  |    /
     *  BB2:
     *  C = phi([BB0, A], [BB1, B])
     */

    BasicBlock* BB = phi->getParent();

    if (vf_info.loop_info->isLoopHeader(BB)) {
        PRINT_HIGH("Loop header at this PHI; propagating input shape");
        z3::expr base = Shape::symbolicExpr(
            vf_info.z3_ctx, phi->getName().str(), sa.base.get_sort().bv_size());
        return Shape::Indexed(base, sa.indices);
    }

    if (sc.isUniform()) {
        PRINT_HIGH("Uniform control flow reconverges at this PHI; "
                   << "propagating input shape");
        return Shape::Indexed(z3::ite(sc.base == 1, sb.base, sa.base),
                              sa.indices);
    }

    PRINT_HIGH("Diverging control flow reconverges at this PHI; "
               << "forcing it to be varying");
    return Shape::Varying();
}

Shape ShapesStep::calculateShapeSelect(SelectInst* select) {
    Shape sc = value_cache.getShape(select->getOperand(0));
    Shape sa = value_cache.getShape(select->getOperand(1));
    Shape sb = value_cache.getShape(select->getOperand(2));

    if (sa.isUnknown() || sb.isUnknown() || sc.isUnknown()) {
        return Shape::Unknown();
    }
    if (!sa.isIndexed() || !sb.isIndexed() || !sc.isIndexed()) {
        return Shape::Varying();
    }

    if (sa.hasConstantBase() && sb.hasConstantBase() && sc.hasConstantBase()) {
        return transformKnownBases(
            [](z3::expr c, z3::expr a, z3::expr b) {
                return z3::ite(c == 1, a, b);
            },
            sc, sa, sb);
    } else if (sc.isUniform()) {
        if (sa.isUniform() && sb.isUniform()) {
            return Shape::Uniform(
                Shape::symbolicExpr(vf_info.z3_ctx, select->getName().str(),
                                    sb.base.get_sort().bv_size()),
                num_lanes);
        } else if (sa.isIndexed() && sb.isIndexed()) {
            z3::expr_vector v(vf_info.z3_ctx);
            for (unsigned i = 0; i < num_lanes; i++) {
                v.push_back(sa.indices[i] == sb.indices[i]);
            }
            z3::expr indices_equal = z3::mk_and(v);
            // If we can statically prove that all the indices are equal
            if (indices_equal.simplify().is_true()) {
                z3::expr base = z3::ite(sc.base == 1, sa.base, sb.base);
                return Shape::Indexed(base, sa.indices);
            }
        }
    }

    return Shape::Varying();
}

Shape ShapesStep::calculateShapeUIToFP(UIToFPInst* uitofp) {
    // Can't track stride for floating point numbers
#if 0
    if (value_cache.getShape(uitofp->getOperand(0)).isUniform()) {
        return Shape::UniformWithoutBase(num_lanes);
    } else {
        return Shape::Varying();
    }
#endif
    return Shape::Varying();
}

Shape ShapesStep::calculateShapeTrunc(TruncInst* trunc) {
    Value* a = trunc->getOperand(0);
    Shape sa = value_cache.getShape(a);
    if (sa.isUnknown()) {
        return Shape::Unknown();
    } else if (sa.isVarying()) {
        return Shape::Varying();
    }
    assert(sa.isIndexed());

    unsigned width = getValueSizeBits(trunc);
    return tryTransform<UnaryShapeTransform>({known_transforms.trunc(width)},
                                             sa);
}

Shape ShapesStep::calculateShapeExt(Instruction* ext, bool is_signed) {
    Value* a = ext->getOperand(0);
    Shape sa = value_cache.getShape(a);
    if (sa.isUnknown()) {
        return Shape::Unknown();
    } else if (sa.isVarying()) {
        return Shape::Varying();
    }
    assert(sa.isIndexed());

    unsigned width = getValueSizeBits(ext);
    if (sa.hasConstantBase()) {
        unsigned ext_bits = width - sa.base.get_sort().bv_size();
        PRINT_HIGH("Adding " << ext_bits << " bits");
        if (is_signed) {
            return transformKnownBases(
                [ext_bits](z3::expr a) { return z3::sext(a, ext_bits); }, sa);
        } else {
            return transformKnownBases(
                [ext_bits](z3::expr a) { return z3::zext(a, ext_bits); }, sa);
        }
    } else {
        if (is_signed) {
            return tryTransform<UnaryShapeTransform>(
                {known_transforms.sext(width)}, sa);
            return Shape::Varying();
        } else {
            return tryTransform<UnaryShapeTransform>(
                {known_transforms.zext(width)}, sa);
        }
    }
}

void ShapesStep::calculateShape(std::unordered_set<Instruction*>& work_queue,
                                Instruction* I, bool allow_overwrite) {
    PRINT_HIGH("");
    PRINT_HIGH("Analyzing shape of " << *I);
    if (I->getType()->isVoidTy()) {
        value_cache.setShape(I, Shape::None());
        return;
    }

    BinaryOperator* binop = dyn_cast<BinaryOperator>(I);
    BitCastInst* bitcast = dyn_cast<BitCastInst>(I);
    CallInst* call = dyn_cast<CallInst>(I);
    GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(I);
    ICmpInst* icmp = dyn_cast<ICmpInst>(I);
    LoadInst* load = dyn_cast<LoadInst>(I);
    PHINode* phi = dyn_cast<PHINode>(I);
    SExtInst* sext = dyn_cast<SExtInst>(I);
    SelectInst* select = dyn_cast<SelectInst>(I);
    TruncInst* trunc = dyn_cast<TruncInst>(I);
    UIToFPInst* uitofp = dyn_cast<UIToFPInst>(I);
    ZExtInst* zext = dyn_cast<ZExtInst>(I);
    ExtractElementInst* extract = dyn_cast<ExtractElementInst>(I);
    InsertElementInst* insert = dyn_cast<InsertElementInst>(I);
    FPToSIInst* fptosi = dyn_cast<FPToSIInst>(I);
    FreezeInst* freeze = dyn_cast<FreezeInst>(I);
    AllocaInst* alloca = dyn_cast<AllocaInst>(I);

    Shape shape = Shape::Unknown();
    if (binop) {
        shape = calculateShapeBinaryOp(binop);
    } else if (bitcast) {
        shape = value_cache.getShape(bitcast->getOperand(0));
    } else if (call) {
        shape = calculateShapeCall(call);
    } else if (gep) {
        shape = calculateShapeGEP(gep);
    } else if (icmp) {
        shape = calculateShapeCmp(icmp);
    } else if (load) {
        shape = calculateShapeLoad(load);
    } else if (phi) {
        shape = calculateShapePHI(phi);
    } else if (select) {
        shape = calculateShapeSelect(select);
    } else if (sext) {
        shape = calculateShapeExt(sext, true);
    } else if (trunc) {
        shape = calculateShapeTrunc(trunc);
    } else if (uitofp) {
        shape = calculateShapeUIToFP(uitofp);
    } else if (zext) {
        shape = calculateShapeExt(zext, false);
    } else if (fptosi) {
        shape = value_cache.getShape(fptosi->getOperand(0));
    } else if (extract || insert) {
        shape = Shape::Varying();
    } else if (freeze) {
        shape = value_cache.getShape(freeze->getOperand(0));
    } else if (alloca && value_cache.has(alloca) &&
               value_cache.getArrayLayoutOpt(alloca)) {
        // We have already calculated shape for this alloca.
        return;
    } else {
        PRINT_HIGH("Don't know how to analyze the shape of " << *I);
    }

    /* TODO: in cases where a scalar value feeds into a varying value,
     * figure out the right policy for deciding whether (and when) to
     * broadcast the value, or to just do it always varying, or to do both
     * simultaneously, or something else. */
    // if (!shape.isVarying()) {
    //    for (User* U : I->users()) {
    //        if (value_cache.has(U) && value_cache.getShape(U).isVarying())
    //        {
    //            PRINT_HIGH("Value feeds into varying value; "
    //                       << "forcing Varying to avoid broadcast");
    //            shape = Shape::Varying();
    //        }
    //    }
    //}

    if (shape.isUnknown()) {
        vf_info.diagnostics.unhandled_shape_opcodes.insert(I->getOpcodeName());
        vf_info.diagnostics.unhandled_shape_insts.push_back(valueString(I));
    }

    // If this changes a pre-existing shape mapping, put any dependents into
    // the work queue (if those dependents are not already varying)
    bool changed = false;
    if (!value_cache.has(I) || value_cache.getShape(I) != shape) {
        changed = true;
    }

    if (changed) {
        for (User* U : I->users()) {
            /* If 'U' is not already in the shapes map, that means we must
             * still be interating through the instruction_order, and
             * haven't gotten to it yet.  Therefore, we will get to it
             * later, and don't have to put it into the work_queue. If 'U'
             * is already in the shapes map, but is already varying, then
             * again we don't have to queue it, because it wouldn't change
             * So the case in which we add 'U' to the work_queue is the case
             * when it's already in the shape map and it's not already
             * varying
             */
            if (value_cache.has(U) && !value_cache.getShape(U).isVarying()) {
                work_queue.insert(cast<Instruction>(U));
                PRINT_HIGH("  Adding user " << *U << " to work queue");
            }
        }
    }

    value_cache.setShape(I, shape, allow_overwrite);
}

void ShapesStep::calulateFinalMemInstMappedShapes() {
    for (Instruction* I : vf_info.instruction_order) {
        LoadInst* load = dyn_cast<LoadInst>(I);
        StoreInst* store = dyn_cast<StoreInst>(I);

        Type* ty;
        MemInstMappedShape ret;
        Shape shape = Shape::None();
        if (load) {
            shape = value_cache.getShape(load->getPointerOperand());
            ty = load->getType();
        } else if (store) {
            shape = value_cache.getShape(store->getPointerOperand());
            ty = store->getValueOperand()->getType();
        } else {
            ret.elem_size = 0;
            ret.mapped_shape = MemInstMappedShape::NONE;
            value_cache.setMemInstMappedShape(I, ret);
            continue;
        }

        TypeSize type_size =
            vf_info.data_layout.getTypeAllocSize(ty->getScalarType());

        ret.elem_size = type_size.getFixedSize();
        if (ty->isVectorTy()) {
            ret.mapped_shape = MemInstMappedShape::ALREADY_PACKED;
        } else if (load && value_cache.getShape(load).global_value) {
            ret.mapped_shape = MemInstMappedShape::GLOBAL_VALUE;
        } else if (shape.isUniform()) {
            ret.mapped_shape = MemInstMappedShape::UNIFORM;
        } else if (shape.isStrided() && shape.getStride() == ret.elem_size) {
            ret.mapped_shape = MemInstMappedShape::PACKED;
        } else if (shape.isGangPacked(ret.elem_size) &&
                   global_opts.scalable_size == 0) {
            ret.mapped_shape = MemInstMappedShape::PACKED_SHUFFLE;
            for (unsigned i = 0; i < shape.indices.size(); i++) {
                if (shape.getIndexAsInt(i) % ret.elem_size != 0) {
                    WARNING(getDebugLocStr(I) +
                            " can't emit PACKED_SHUFFLE because indices "
                            "are not a multiple of the element size, emitting "
                            "GATHER_SCATTER instead");
                    ret.indices.clear();
                    ret.mapped_shape = MemInstMappedShape::GATHER_SCATTER;
                    break;
                }
                ret.indices.push_back(
                    static_cast<int>(shape.getIndexAsInt(i) / ret.elem_size));
            }
        } else {
            ret.mapped_shape = MemInstMappedShape::GATHER_SCATTER;
        }
        value_cache.setMemInstMappedShape(I, ret);
    }
}

void ShapesStep::printShapes() {
    PRINT_LOW("Final shapes for: " << llvm::demangle(vf_info.vfabi.scalar_name)
                                   << ": gang size = " << vf_info.vfabi.vlen);

    for (BasicBlock& BB : *vf_info.VF) {
        PRINT_LOW("Basic block " << BB.getName() << ":");
        for (Instruction& I : BB) {
            MemInstMappedShape mem_instr_shape =
                value_cache.getMemInstMappedShape(&I);

            std::string s;
            if (mem_instr_shape.mapped_shape != MemInstMappedShape::NONE) {
                s += "; " + mem_instr_shape.toString();
            }
            PRINT_LOW(I << "; " << value_cache.getShape(&I).toString() << s);
        }
    }
}

void ShapesStep::calculate() {
    PRINT_MID("\n");
    PRINT_LOW("Calculating shapes for:" << vf_info.VF->getName());
    PRINT_HIGH("Function is:\n" << *vf_info.VF);
    PRINT_MID("");

    std::unordered_set<Instruction*> work_queue;

    unsigned i = 0;
    for (; i < vf_info.vfabi.parameters.size(); i++) {
        auto p = vf_info.vfabi.parameters[i];
        auto arg = vf_info.VF->getArg(i);
        if (p.is_varying) {
            value_cache.setShape(vf_info.VF->getArg(i), Shape::Varying());
        } else {
            unsigned width = getValueSizeBits(arg);
            std::string name = value_cache.getConstName(arg);
            z3::expr base =
                Shape::symbolicExpr(vf_info.solver, name, width, p.alignment);
            value_cache.setShape(arg,
                                 Shape::Strided(base, p.stride, num_lanes));
        }
    }

    // Make sure the order for the extra arguments matches the calling
    // convention used elsewhere
    if (vf_info.vfabi.mask) {
        value_cache.setShape(vf_info.VF->getArg(i), Shape::Varying());
        i++;
    }

    for (GlobalValue& v : vf_info.mod->globals()) {
        unsigned width = getValueSizeBits(&v);
        std::string name = value_cache.getConstName(&v);
        z3::expr base = Shape::symbolicExpr(vf_info.solver, name, width);
        value_cache.setShape(&v, Shape::Uniform(base, num_lanes));
        shape_constants[name] = &v;
    }

    arrayLayoutOpt();

    for (Instruction* I : vf_info.instruction_order) {
        calculateShape(work_queue, I);
    }

    PRINT_HIGH("Iterating through work queue");

    while (!work_queue.empty()) {
        auto it = work_queue.begin();
        calculateShape(work_queue, *it, true);
        work_queue.erase(it);
    }

    calulateFinalMemInstMappedShapes();

    DEBUG_MID(printShapes());
}

unsigned ShapesStep::getValueSizeBits(Value* v) {
    Type* ty = v->getType()->getScalarType();
    IntegerType* ity = dyn_cast<IntegerType>(ty);
    if (ity) {
        return ity->getBitWidth();
    } else {
        return getValueSizeBytes(v) * 8;
    }
}

unsigned ShapesStep::getValueSizeBytes(Value* v) {
    Type* ty = v->getType()->getScalarType();
    return vf_info.data_layout.getTypeAllocSize(ty).getFixedSize();
}

unsigned ShapesStep::getBaseValueSizeBytes(Value* v) {
    Type* ty = v->getType()->getScalarType();
    Type* old_ty = nullptr;
    while (ty != old_ty) {
        old_ty = ty;
        PointerType* pty = dyn_cast<PointerType>(ty);
        if (pty) {
            ty = pty->getNonOpaquePointerElementType();
        }
        ArrayType* aty = dyn_cast<ArrayType>(ty);
        if (aty) {
            ty = aty->getElementType();
        }
    }
    PRINT_HIGH("Getting size of type " << *ty);
    return vf_info.data_layout.getTypeAllocSize(ty).getFixedSize();
}

GlobalValue* ShapesStep::getGlobalValueFromExpr(z3::expr base) {
    auto it = shape_constants.find(base.simplify().to_string());
    if (it != shape_constants.end()) {
        PRINT_HIGH("Found global value " << it->first);
        return it->second;
    }
    return nullptr;
}

ShapesStep::GlobalValuePlusOffset ShapesStep::getGlobalValuePlusOffsetFromExpr(
    z3::expr base) {
    z3::expr e = base.simplify();

    if (e.is_const()) {
        GlobalValue* gv = getGlobalValueFromExpr(e);
        if (gv) {
            return {gv, 0};
        }
        PRINT_HIGH("Constant is not a known GlobalValue");
        return {nullptr, 0};
    }

    if (e.is_app()) {
        z3::func_decl decl = e.decl();
        PRINT_HIGH("Base is function application with function decl "
                   << decl.to_string());

        if (decl.decl_kind() == Z3_OP_BADD) {
            PRINT_HIGH("Found bvadd");
            GlobalValuePlusOffset gv_plus_offset{nullptr, 0};
            for (unsigned i = 0; i < e.num_args(); i++) {
                z3::expr arg = e.arg(i);
                PRINT_HIGH("Found bvadd arg " << arg.to_string());

                GlobalValue* gv = getGlobalValueFromExpr(arg);
                if (gv) {
                    if (gv_plus_offset.gv) {
                        PRINT_HIGH("Two symbols in expression");
                        return {nullptr, 0};
                    }
                    gv_plus_offset.gv = gv;
                }

                uint64_t val;
                if (arg.is_numeral_u64(val)) {
                    ASSERT(
                        gv_plus_offset.offset == 0,
                        "multiple constants in expression? " << e.to_string());
                    gv_plus_offset.offset = val;
                }
            }
            return gv_plus_offset;
        } else {
            PRINT_HIGH("Base function is not bvadd");
            return {nullptr, 0};
        }
    }

    PRINT_HIGH("Base is some other type of expression");
    return {nullptr, 0};
}

}  // namespace ps
