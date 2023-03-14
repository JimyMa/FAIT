#pragma once

#include <torch/csrc/jit/tensorexpr/ir.h>

#include "util/types.h"

namespace torch {
namespace jit {
namespace tensorexpr {

template <class T>
inline ExprHandle getScalarExpr(
    Value* value, const std::unordered_map<Value*, ExprHandle>& valueToExpr) {
  auto cnst = constant_as<T>(value);
  if (cnst)
    return ExprHandle(getImmediateByType<T>(GetScalarType<T>::result, *cnst));
  else
    return valueToExpr.at(value);
}

template <class T>
inline std::vector<ExprHandle> getExprList(
    Value* value, const std::unordered_map<Value*, ExprHandle>& valueToExpr) {
  TORCH_CHECK(value->type()->kind() == TypeKind::ListType);
  std::vector<ExprHandle> result;
  auto node = value->node();
  if (node->kind() == prim::Constant) {
    auto cnst = toIValue(value);
    for (auto& elem : cnst->toListRef())
      result.emplace_back(
          getImmediateByType<T>(GetScalarType<T>::result, elem.to<T>()));
  } else if (node->kind() == prim::ListConstruct) {
    for (auto input : node->inputs())
      result.push_back(getScalarExpr<T>(input, valueToExpr));
  } else {
    TORCH_CHECK(false);
  }
  return result;
}

#define GET_BUF_AT(idx) \
  BufHandle(valueToExpr.at(node->input(idx)).AsNode<Buf>())

#define GET_CONST_AT(idx, type) *constant_as<type>(node->input(idx))
#define GET_INT_CONST_AT(idx) GET_CONST_AT(idx, int64_t)

#define GET_SCALAR_EXPR_AT(idx, type) \
  getScalarExpr<type>(node->input(idx), valueToExpr);
#define GET_INT_SCALAR_EXPR_AT(idx) GET_SCALAR_EXPR_AT(idx, int64_t)

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch