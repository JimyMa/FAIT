#pragma once

#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/hash_provider.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class ExprHasher {
 public:
  size_t operator()(const ExprPtr &expr) const {
    return provider.hash(expr)._h;
  }

 private:
  mutable HashProvider provider;
};

struct ExprEq {
  bool operator()(const ExprPtr &lhs, const ExprPtr &rhs) const {
    return std::to_string(lhs) == std::to_string(rhs);
  }
};

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch
