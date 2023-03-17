#pragma once

#include <torch/csrc/jit/tensorexpr/expr.h>

namespace torch {
namespace jit {
namespace tensorexpr {

enum IRNodeTypeExtend {
  kEnd = IRNodeType::kOther,
};

class TORCH_API Tuple : public ExprNode<Tuple> {
 public:
  static ExprHandle make(const std::vector<ExprPtr> elements, Dtype dtype) {
    return ExprHandle(alloc<Tuple>(elements, dtype));
  }
  static ExprHandle make(Dtype dtype) {
    return ExprHandle(alloc<Tuple>(std::vector<ExprPtr>(), dtype));
  }

  // TODO: unique_name
  const std::vector<ExprPtr>& elements() const { return elements_; }

  Tuple(std::vector<ExprPtr> elements, Dtype dtype)
      : ExprNodeBase(dtype, kOther), elements_(std::move(elements)) {}

 private:
  std::vector<ExprPtr> elements_;
};

template <>
void ExprNode<Tuple, Expr>::accept(IRVisitor* mutator) {}

template <>
ExprPtr ExprNode<Tuple, Expr>::accept_mutator(IRMutator* mutator) {
  return nullptr;
}

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch
