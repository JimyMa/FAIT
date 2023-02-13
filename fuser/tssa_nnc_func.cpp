//
// Created by jimyma on 2/10/23.
//

#include "fuser/tssa_nnc_func.h"


namespace torch {
namespace jit {
namespace tensorexpr {

Tensor computeAssign(
        const std::vector<ArgValue>& inputValues,
        const std::vector<ExprHandle>& outputShape,
        const std::vector<ExprHandle>& outputStrides,
        const c10::optional<ScalarType>& outputType,
        at::Device device) {
  return Compute(
          "assign",
          outputShape,
          outputStrides,
          [inputValues, outputType](const std::vector<VarHandle>& axes) {
            return c10::get_if<BufHandle>(&inputValues[0])->load(axes);
          });
}

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch


