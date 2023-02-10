#pragma once

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/ir.h>

#include "refine_types.h"
#include "util/traits.h"

namespace torch {
namespace jit {

using ShapeVec = std::vector<c10::optional<int64_t>>;

template <class Stream>
inline void dump(Stream &stream) {}

template <class Stream, class Arg, class... Args>
inline void dump(Stream &stream, Arg &&arg, Args &&...args) {
    stream << arg;
    dump(stream, std::forward<Args>(args)...);
}

template <class Error, class... Args>
inline Error error(Args &&...args) {
    std::stringstream ss;
    dump(ss, std::forward<Args>(args)...);
    return Error(ss.str(), c10::get_backtrace(1));
}

template <class... Args>
inline c10::TypeError typeError(Args &&...args) {
    return error<c10::TypeError>(std::forward<Args>(args)...);
}

inline bool anyIsNone() { return false; }

template <class T, class... Opts>
inline bool anyIsNone(const c10::optional<T> &opt, const Opts &...opts) {
    if (!opt) return true;
    return anyIsNone(opts...);
}

template <class T, class F, class... Opts>
inline c10::optional<T> tryApply(const F &func, const Opts &...opts) {
    if (anyIsNone(opts...)) return c10::nullopt;
    return func(*opts...);
}

template <class Out, class In, class F>
inline c10::optional<Out> mapOpt(const c10::optional<In> &from, F &&mapFunc) {
    if (from)
        return mapFunc(*from);
    else
        return c10::nullopt;
}

template <class T>
inline c10::optional<T> joinOpt(const c10::optional<T> &accum,
                                const c10::optional<T> &newVal) {
    if (accum)
        return accum;
    else
        return newVal;
}

inline TypePtr getRefinedType(Value *value, ValueTypeMap &refinedTypes) {
    if (refinedTypes.count(value))
        return refinedTypes[value];
    else
        return value->type();
}

inline bool isTensor(Value *v) {
    return v->type()->kind() == TypeKind::TensorType;
}

inline auto getShape(const TypePtr &tensorTy) {
    TORCH_INTERNAL_ASSERT(tensorTy->kind() == TypeKind::TensorType);
    return tensorTy->cast<TensorType>()->sizes().sizes();
}

inline auto getRank(const TypePtr &tensorTy) {
    TORCH_INTERNAL_ASSERT(tensorTy->kind() == TypeKind::TensorType);
    return tensorTy->cast<TensorType>()->sizes().size();
}

inline c10::SymbolicShape getRankedShape(size_t rank) {
    return c10::optional<size_t>(rank);
}

inline c10::optional<size_t> getListLen(Value *list,
                                        ValueTypeMap &refinedTypes) {
    auto listTy = getRefinedType(list, refinedTypes);
    if (listTy->kind() == TypeKind::TupleType)
        return listTy->cast<TupleType>()->elements().size();
    else
        return c10::nullopt;
}

inline c10::optional<std::vector<c10::optional<int64_t>>> getIntList(
    Value *value) {
    TORCH_INTERNAL_ASSERT(*value->type() == *ListType::create(IntType::get()));
    if (isMutated(value)) return c10::nullopt;
    auto node = value->node();
    switch (node->kind()) {
        case prim::Constant: {
            auto cnstList = toIValue(value)->toIntVector();
            std::vector<c10::optional<int64_t>> retList;
            for (auto c : cnstList) retList.emplace_back(c);
            return retList;
        };

        case prim::ListConstruct: {
            std::vector<c10::optional<int64_t>> retList;
            for (auto input : node->inputs()) {
                auto ival = toIValue(input);
                if (ival)
                    retList.push_back(ival->toInt());
                else
                    retList.push_back(c10::nullopt);
            }
            return retList;
        }

        case aten::size: {
            auto tensor = node->input(0);
            return tensor->type()->cast<TensorType>()->sizes().sizes();
        }
    }
    return c10::nullopt;
}

inline TypePtr getElementType(const TypePtr &type, size_t index) {
    switch (type->kind()) {
        case TypeKind::ListType:
            return type->cast<ListType>()->getElementType();

        case TypeKind::TupleType:
            return type->cast<TupleType>()->elements().at(index);

        default:
            TORCH_INTERNAL_ASSERT(false, "Unreachable");
    }

    return nullptr;
}

inline TypePtr getUnifiedElementType(const TypePtr &type) {
    switch (type->kind()) {
        case TypeKind::ListType:
            return type->cast<ListType>()->getElementType();

        case TypeKind::TupleType: {
            auto elemTy = at::unifyTypeList(type->cast<TupleType>()->elements(),
                                            std::cout);
            if (!elemTy) throw typeError("Cannot unify elements in ", *type);
            return *elemTy;
        }

        default:
            TORCH_INTERNAL_ASSERT(false, "Unreachable");
    }

    return nullptr;
}

template <class AttrType, class GetFunc,
          class CombineFunc = decltype(joinOpt<AttrType>)>
inline c10::optional<AttrType> accumAttrFromElements(
    const TypePtr &listTy, GetFunc &&getFunc,
    CombineFunc &&combineFunc = joinOpt<AttrType>,
    const c10::optional<AttrType> &initVal = c10::nullopt) {
    switch (listTy->kind()) {
        case TypeKind::ListType:
            return getFunc(listTy->cast<ListType>()->getElementType());

        case TypeKind::TupleType: {
            auto elemTypes = listTy->cast<TupleType>()->elements();
            auto result = initVal;
            for (auto elemTy : elemTypes)
                result = combineFunc(std::move(result), getFunc(elemTy));
            return std::move(result);
        }

        default:
            TORCH_INTERNAL_ASSERT(false, "Unreachable");
    }
    return c10::nullopt;
}

}  // namespace jit
}  // namespace torch
