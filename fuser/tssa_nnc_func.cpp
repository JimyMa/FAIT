//
// Created by jimyma on 2/10/23.
//

#include "fuser/tssa_nnc_func.h"
#include <c10/util/variant.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>
#include <torch/csrc/jit/tensorexpr/types.h>
#include <utility>


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
          [inputValues, outputType](const std::vector<VarHandle>& axes) {
            return c10::get_if<BufHandle>(&inputValues[0])->load(axes);
          });
}

Tensor computeSlice(
  const std::vector<ArgValue>& inputValues,
        const std::vector<ExprHandle>& outputShape,
        const std::vector<ExprHandle>& outputStrides,
        const c10::optional<ScalarType>& outputType,
        at::Device device) {
  return Compute(
          "slice",
          outputShape,
          [inputValues, outputType](const std::vector<VarHandle>& axes) {
            auto src = c10::get_if<BufHandle>(&inputValues[0]);
            auto dim = c10::get_if<int64_t>(&inputValues[1]);
            
            int64_t start;
            if (c10::get_if<ArgNone>(&inputValues[2])) {
              start = 0;
            } else {
              start = *c10::get_if<int64_t>(&inputValues[2]);
            }

            int64_t step = *c10::get_if<int64_t>(&inputValues[4]);

            std::vector<ExprHandle> output_idx;
            for (auto axis : axes) {
              output_idx.push_back(axis);
            }
            output_idx[*dim] = LongImm::make(start) + LongImm::make(step) * output_idx[*dim];
            return src->load(output_idx);
          });
}

Tensor computeSelect(
        const std::vector<ArgValue>& inputValues,
        const std::vector<ExprHandle>& outputShape,
        const std::vector<ExprHandle>& outputStrides,
        const c10::optional<ScalarType>& outputType,
        at::Device device) {
  return Compute(
          "select",
          outputShape,
          [inputValues, outputType](const std::vector<VarHandle>& axes) {
            auto src = c10::get_if<BufHandle>(&inputValues[0]);
            auto dim = c10::get_if<int64_t>(&inputValues[1]);
            auto idx = c10::get_if<int64_t>(&inputValues[2]);
            std::vector<ExprHandle> output_idx;
            for (auto axis : axes) {
              output_idx.push_back(axis);
            }
            output_idx.insert(output_idx.begin() + *dim, LongImm::make(*idx));
            return src->load(output_idx);
          });
}

std::vector<ExprHandle>
computePointwiseShape(std::vector<ArgValue> input_args) {
  return c10::get_if<BufHandle>(&input_args[0])->dims();
}

std::vector<ExprHandle>
computeSelectShape(std::vector<ArgValue> input_args) {
  // TODO: DIM Must be a constant
  auto src = c10::get_if<BufHandle>(&input_args[0]);
  auto dim = c10::get_if<int64_t>(&input_args[1]);
  if (!dim) {
    std::cout << "[ERROR] Must be a constant by now!" << std::endl;
    throw unsupported_dtype("[ERROR] Must be a constant by now!");
  }

  auto result = src->dims();
  result.erase(result.begin() + *dim);
  std::cout << "size: " << src->dims().size() << std::endl;
  std::cout << "size: " << result.size() << std::endl;
  return result;
}

std::vector<ExprHandle>
computeSliceShape(std::vector<ArgValue> input_args) {
  auto src = c10::get_if<BufHandle>(&input_args[0]);
  auto dim = *c10::get_if<int64_t>(&input_args[1]);
  
  int64_t start;
  if (c10::get_if<ArgNone>(&input_args[2])) {
    start = 0;
  } else {
    start = *c10::get_if<int64_t>(&input_args[2]);
  }
  ExprHandle start_expr = LongImm::make(start);

  ExprHandle end_expr;
  if (c10::get_if<ArgNone>(&input_args[3])) {
    end_expr = src->dim(dim);
  } else {
    end_expr = LongImm::make(*c10::get_if<int64_t>(&input_args[3]));
  }

  
  int64_t step = *c10::get_if<int64_t>(&input_args[4]);
  ExprHandle step_expr = LongImm::make(step);
  std::vector<ExprHandle> result = src->dims();
  result[dim] = (end_expr - start_expr) / step_expr;
  return result;
}


}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch


