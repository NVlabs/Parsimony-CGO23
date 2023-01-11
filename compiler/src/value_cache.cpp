/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "value_cache.h"

#include <llvm/IR/IRBuilder.h>

#include "broadcast.h"
#include "utils.h"
#include "vectorize.h"

using namespace llvm;

namespace ps {

unsigned value_cache_verbosity_level;
[[maybe_unused]] static unsigned& verbosity_level = value_cache_verbosity_level;

bool ValueCache::has(Value* value) const {
    return entries.find(value) != entries.end();
}

void ValueCache::setToBeDeleted(Value* value) {
    auto it = entries.find(value);
    if (it == entries.end()) {
        FATAL("Value cache entry for " << *value << " does not exist\n");
    }
    it->second.to_be_deleted = true;
}

void ValueCache::setArrayLayoutOpt(Value* value) {
    auto it = entries.find(value);
    if (it == entries.end()) {
        FATAL("Value cache entry for " << *value << " does not exist\n");
    }
    it->second.arrayLayoutOpt = true;
}

void ValueCache::setVectorValue(Value* value, Value* vector_value) {
    auto it = entries.find(value);
    if (it == entries.end()) {
        FATAL("Value cache entry for " << *value << " does not exist\n");
    }
    if (it->second.vector_value) {
        FATAL("Value cache entry for " << *value
                                       << " already has a vector value\n");
    }
    if (vector_value) {
        PRINT_HIGH("Setting vector value for " << *value << " to "
                                               << *vector_value);
        ASSERT(vector_value->getType()->isVectorTy(),
               "Setting vector value of " << *value << " to scalar value "
                                          << *vector_value);
    } else {
        PRINT_HIGH("Setting vector value for " << *value << " to null");
    }
    it->second.vector_value = vector_value;
}

void ValueCache::setScalarValue(Value* value, Value* scalar_value) {
    auto it = entries.find(value);
    if (it == entries.end()) {
        FATAL("Value cache entry for " << *value << " does not exist\n");
    }
    if (it->second.scalar_value) {
        FATAL("Value cache entry for " << *value
                                       << " already has a scalar value\n");
    }
    if (scalar_value) {
        PRINT_HIGH("Setting scalar value for " << *value << " to "
                                               << *scalar_value);
        ASSERT(!scalar_value->getType()->isVectorTy(),
               "Setting scalar value of " << *value << " to non-scalar value "
                                          << *scalar_value);
    } else {
        PRINT_HIGH("Setting scalar value for " << *value << " to null");
    }
    it->second.scalar_value = scalar_value;
}

Value* ValueCache::getScalarValue(Value* value) {
    if (get(value).scalar_value) {
        return get(value).scalar_value;
    }
    return value;
}

std::string ValueCache::getConstName(llvm::Value* value) {
    if (value->getName().empty()) {
        return unknown_const_name_string +
               std::to_string(unknown_const_name_counter++);
    }
    return value->getName().str();
}

Value* ValueCache::getVectorValue(Value* value) {
    // Check for value_cache hit
    auto it = entries.find(value);
    if (it != entries.end()) {
        Value* ret = it->second.vector_value;
        if (ret) {
            PRINT_HIGH("Using cached vector_value " << *ret);
            return ret;
        }
    }
    // if the value was already vector just return it
    // some vector values may not even be in the value cache so it is
    // not safe to check for the shape
    if (value->getType()->isVectorTy()) {
        PRINT_HIGH("Value is already vector: " << *value);
        return value;
    }

    Type* ty = value->getType();
    Shape shape = getShape(value);
    PRINT_HIGH("Vectorizing " << *value << "; Type " << *ty << "; "
                              << shape.toString());

    // Broadcast or generate vector constant depending on what type of LLVM
    // value 'value' is
    Type* i64 = Type::getInt64Ty(vf_info->ctx);

    StringRef name = value->getName();
    assert(!shape.isVarying());
    assert(shape.isIndexed());
    Type* idx_ty = ty;

    // Get a scalar value if there is already one
    Value* val = getScalarValue(value);
    assert(ty = val->getType());

    // create builder and select insert point
    Instruction* I = dyn_cast<Instruction>(val);
    IRBuilder<> builder(vf_info->ctx);
    if (I) {
        if (isa<PHINode>(I->getNextNode())) {
            builder.SetInsertPoint(
                I->getParent()->getFirstNonPHIOrDbgOrLifetime());
        } else {
            builder.SetInsertPoint(I->getNextNode());
        }
    } else {
        builder.SetInsertPoint(VF->begin()->getFirstNonPHIOrDbgOrLifetime());
    }

    if (ty->isPointerTy()) {
        val = builder.CreatePtrToInt(val, i64, name + ".");
        idx_ty = i64;
    }
    Value* ret;
    if (shape.hasConstantBase()) {
        uint64_t base = shape.getConstantBase();
        std::vector<Constant*> idxs;
        for (int64_t i = 0; i < num_lanes; i++) {
            idxs.push_back(
                ConstantInt::get(idx_ty, base + shape.getIndexAsInt(i)));
        }
        ret = genConstVect(ConstantVector::get(idxs), builder);

    } else if (shape.isUniform()) {
        ret = builder.CreateVectorSplat(getElementCount(num_lanes), val,
                                        name + ".");
    } else {
        Value* bcast = builder.CreateVectorSplat(getElementCount(num_lanes),
                                                 val, name + ".");
        std::vector<Constant*> idxs;
        for (int64_t i = 0; i < num_lanes; i++) {
            idxs.push_back(ConstantInt::get(idx_ty, shape.getIndexAsInt(i)));
        }
        Value* vidx = genConstVect(ConstantVector::get(idxs), builder);
        ret = builder.CreateAdd(bcast, vidx, name + ".");
    }

    if (ty->isPointerTy()) {
        Type* vty = vf_info->vectorizeType(ty);
        ret = builder.CreateIntToPtr(ret, vty, name + ".");
    }

    setVectorValue(value, ret);
    PRINT_HIGH("Vectorized value result is " << *ret);
    return ret;
}

Value* ValueCache::genConstVect(Constant* C, IRBuilder<>& builder) {
    if (global_opts.scalable_size) {
        GlobalVariable* ptr_C =
            new GlobalVariable(*vf_info->mod, C->getType(), true,
                               GlobalValue::InternalLinkage, 0, "const");
        ptr_C->setInitializer(C);

        Type* vty = VectorType::get(C->getType()->getScalarType(),
                                    getElementCount(num_lanes));
        Type* vpty = PointerType::get(vty, 0);
        Value* ptr_C_cast = builder.CreateBitCast(ptr_C, vpty);
        return builder.CreateLoad(vty, ptr_C_cast);
    } else {
        return dyn_cast<Value>(C);
    }
}

void ValueCache::setShape(Value* value, Shape shape, bool overwrite) {
    PRINT_HIGH("Setting shape of " << *value << " to " << shape.toString());
    auto it = entries.find(value);
    if (it != entries.end()) {
        if (!overwrite && !it->second.shape.isNone()) {
            FATAL("Overwriting shape for " << *value);
        }

        ValueCacheEntry& entry = get(value);
        entry.shape = shape;
    } else {
        entries.insert(std::make_pair(value, ValueCacheEntry(nullptr, shape)));
    }
}

Shape ValueCache::getShape(Value* value) { return get(value).shape; }

bool ValueCache::getArrayLayoutOpt(Value* value) {
    return get(value).arrayLayoutOpt;
}

MemInstMappedShape ValueCache::getMemInstMappedShape(Instruction* inst) {
    auto it = entries.find(inst);
    assert(it != entries.end());
    return it->second.minst_mapping;
}

void ValueCache::setMemInstMappedShape(Instruction* inst,
                                       MemInstMappedShape minst_mapping) {
    auto it = entries.find(inst);
    assert(it != entries.end());
    it->second.minst_mapping = minst_mapping;
}

ValueCache::ValueCacheEntry& ValueCache::get(Value* value) {
    auto it = entries.find(value);
    if (it == entries.end()) {
        z3::context& ctx = vf_info->z3_ctx;

        /* FIXME deduplicate this with ShapesStep::getValueSizeBits() */
        Type* ty = value->getType()->getScalarType();
        IntegerType* ity = dyn_cast<IntegerType>(ty);
        unsigned width;
        if (ity) {
            width = ity->getBitWidth();
        } else {
            width =
                vf_info->data_layout.getTypeAllocSize(ty).getFixedSize() * 8;
        }

        Constant* C = dyn_cast<Constant>(value);
        ConstantInt* Cint = dyn_cast<ConstantInt>(value);
        if (Cint) {
            int64_t cval = Cint->getValue().getZExtValue();
            setShape(Cint, Shape::Uniform(Shape::constantExpr(ctx, cval, width),
                                          num_lanes));
            return entries.find(Cint)->second;
        } else if (C) {
            setShape(C, Shape::Uniform(
                            Shape::symbolicExpr(ctx, getConstName(C), width),
                            num_lanes));
            return entries.find(C)->second;
        }

        FATAL("Could not find shape for value " << *value);
    }
    return it->second;
}

void ValueCache::deleteObsoletedInsts() {
    for (auto& i : entries) {
        Instruction* I = dyn_cast<Instruction>(i.first);
        ValueCacheEntry& entry = i.second;

        if (entry.to_be_deleted && !entry.already_deleted) {
            deleteInst(I, 0);
        }
    }
}

void ValueCache::deleteInst(Instruction* I, unsigned prefix) {
    std::string p;
    for (unsigned i = 0; i < prefix; i++) {
        p += "  ";
    }

    PRINT_HIGH(p << "Trying to delete " << *I);
    assert(entries.find(I) != entries.end());
    if (entries.at(I).already_deleted) {
        return;
    }

    PRINT_HIGH(p << "Deleting " << *I);
    entries.at(I).already_deleted = true;

    while (I->user_begin() != I->user_end()) {
        auto u = *I->user_begin();
        Instruction* ui = cast<Instruction>(u);
        assert(ui != I);
        deleteInst(ui, prefix + 1);
    }

    // unlink and delete (removeFromParent() just unlinks)
    I->eraseFromParent();
}

}  // namespace ps
