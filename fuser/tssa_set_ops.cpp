#include "tssa_set_ops.h"

#include <torch/csrc/jit/runtime/operator.h>

namespace torch {
namespace jit {

static bool registered = false;

RegisterOperators registerTssaSetOps() {
  if (registered) return {};
  registered = true;
  return RegisterOperators()
      .op("tssa::SelectSet(Tensor self, Tensor src, int dim, int index) -> "
          "Tensor")
      .op("tssa::SliceSet(Tensor self, Tensor src, int dim=0, SymInt? "
          "start=None, SymInt? end=None, SymInt step=1) -> Tensor")
      .op("tssa::Assign(Tensor self, Tensor src) -> Tensor");
}

}  // namespace jit
}  // namespace torch
