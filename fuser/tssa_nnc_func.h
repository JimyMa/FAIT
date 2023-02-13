//
// Created by jimyma on 2/10/23.
//

#ifndef LONG_TAIL_TSSA_NNC_FUNC_H
#define LONG_TAIL_TSSA_NNC_FUNC_H
#include "torch/csrc/jit/tensorexpr/lowerings.h"

namespace torch {
namespace jit {
namespace tensorexpr {

Tensor computeAssign(
        const std::vector<ArgValue>& inputValues,
        const std::vector<ExprHandle>& outputShape,
        const std::vector<ExprHandle>& outputStrides,
        const c10::optional<ScalarType>& outputType,
        at::Device device);

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch

#endif //LONG_TAIL_TSSA_NNC_FUNC_H
