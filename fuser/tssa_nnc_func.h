//
// Created by jimyma on 2/10/23.
//

#ifndef LONG_TAIL_TSSA_NNC_FUNC_H
#define LONG_TAIL_TSSA_NNC_FUNC_H
#include "torch/csrc/jit/tensorexpr/lowerings.h"
#include <torch/csrc/jit/tensorexpr/expr.h>

namespace torch {
namespace jit {
namespace tensorexpr {

Tensor computeAssign(
        const std::vector<ArgValue>& inputValues,
        const std::vector<ExprHandle>& outputShape,
        const std::vector<ExprHandle>& outputStrides,
        const c10::optional<ScalarType>& outputType,
        at::Device device);

Tensor computeSelect(
        const std::vector<ArgValue>& inputValues,
        const std::vector<ExprHandle>& outputShape,
        const std::vector<ExprHandle>& outputStrides,
        const c10::optional<ScalarType>& outputType,
        at::Device device);

Tensor computeSlice(
        const std::vector<ArgValue>& inputValues,
        const std::vector<ExprHandle>& outputShape,
        const std::vector<ExprHandle>& outputStrides,
        const c10::optional<ScalarType>& outputType,
        at::Device device);

Tensor computeAssign(
        const std::vector<ArgValue>& inputValues,
        const std::vector<ExprHandle>& outputShape,
        const std::vector<ExprHandle>& outputStrides,
        const c10::optional<ScalarType>& outputType,
        at::Device device);

Tensor computeSliceSet(
        const std::vector<ArgValue>& inputValues,
        const std::vector<ExprHandle>& outputShape,
        const std::vector<ExprHandle>& outputStrides,
        const c10::optional<ScalarType>& outputType,
        at::Device device);

Tensor computeSelectSet(
        const std::vector<ArgValue>& inputValues,
        const std::vector<ExprHandle>& outputShape,
        const std::vector<ExprHandle>& outputStrides,
        const c10::optional<ScalarType>& outputType,
        at::Device device);


std::vector<ExprHandle> computePointwiseShape(std::vector<ArgValue> outputShape);
std::vector<ExprHandle> computeSelectShape(std::vector<ArgValue> outputShape);
std::vector<ExprHandle> computeSliceShape(std::vector<ArgValue> outputShape);
std::vector<ExprHandle> computeAssignShape(std::vector<ArgValue> outputShape);
std::vector<ExprHandle> computePermuteShape(std::vector<ArgValue> input_args);
std::vector<ExprHandle> computeReshapeShape(std::vector<ArgValue> input_args);

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch

#endif //LONG_TAIL_TSSA_NNC_FUNC_H
