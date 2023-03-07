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

Tensor computeAssign(const std::vector<ArgValue>& inputValues,
                     const std::vector<ExprHandle>& outputShape,
                     const std::vector<ExprHandle>& outputStrides,
                     const c10::optional<ScalarType>& outputType,
                     at::Device device) {
  return Compute("assign", outputShape,
                 [inputValues, outputType](const std::vector<VarHandle>& axes) {
                   return c10::get_if<BufHandle>(&inputValues[1])->load(axes);
                 });
}

Tensor computeSelect(const std::vector<ArgValue>& inputValues,
                     const std::vector<ExprHandle>& outputShape,
                     const std::vector<ExprHandle>& outputStrides,
                     const c10::optional<ScalarType>& outputType,
                     at::Device device) {
  return Compute(
      "select", outputShape, [&](const std::vector<VarHandle>& axes) {
        auto src = c10::get_if<BufHandle>(&inputValues[0]);
        auto dim = *c10::get_if<int64_t>(&inputValues[1]);
        dim = dim == -1 ? src->dims().size() - 1 : dim;
        auto idx = c10::get_if<int64_t>(&inputValues[2]);
        std::vector<ExprHandle> output_idx;
        for (auto axis : axes) {
          output_idx.push_back(axis);
        }
        output_idx.insert(output_idx.begin() + dim, LongImm::make(*idx));
        return src->load(output_idx);
      });
}
Tensor computeSelectSet(const std::vector<ArgValue>& inputValues,
                        const std::vector<ExprHandle>& outputShape,
                        const std::vector<ExprHandle>& outputStrides,
                        const c10::optional<ScalarType>& outputType,
                        at::Device device) {
  return Compute(
      "select_set", outputShape, [&](const std::vector<VarHandle>& axes) {
        auto src = c10::get_if<BufHandle>(&inputValues[0]);
        auto select_setter = c10::get_if<BufHandle>(&inputValues[1]);
        auto dim = *c10::get_if<int64_t>(&inputValues[2]);
        dim = dim == -1 ? src->dims().size() - 1 : dim;
        auto idx = c10::get_if<int64_t>(&inputValues[3]);
        std::vector<ExprHandle> setter_idx;
        for (auto axis : axes) {
          setter_idx.push_back(axis);
        }
        setter_idx.erase(setter_idx.begin() + dim);

        auto cond = CompareSelect::make(
            axes[dim], LongImm::make(*idx), select_setter->load(setter_idx),
            src->load(axes), CompareSelectOperation::kEQ);

        return cond;
      });
}

Tensor computeSlice(const std::vector<ArgValue>& inputValues,
                    const std::vector<ExprHandle>& outputShape,
                    const std::vector<ExprHandle>& outputStrides,
                    const c10::optional<ScalarType>& outputType,
                    at::Device device) {
  return Compute("slice", outputShape, [&](const std::vector<VarHandle>& axes) {
    auto src = c10::get_if<BufHandle>(&inputValues[0]);
    auto dim = *c10::get_if<int64_t>(&inputValues[1]);
    if (dim == -1) {
      dim = src->dims().size() - 1;
    }
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
    output_idx[dim] =
        LongImm::make(start) + LongImm::make(step) * output_idx[dim];
    return src->load(output_idx);
  });
}

Tensor computeSliceSet(const std::vector<ArgValue>& inputValues,
                       const std::vector<ExprHandle>& outputShape,
                       const std::vector<ExprHandle>& outputStrides,
                       const c10::optional<ScalarType>& outputType,
                       at::Device device) {
  return Compute(
      "slice_set", outputShape, [&](const std::vector<VarHandle>& axes) {
        auto src = c10::get_if<BufHandle>(&inputValues[0]);
        auto slice_setter = c10::get_if<BufHandle>(&inputValues[1]);

        auto dim = *c10::get_if<int64_t>(&inputValues[2]);
        if (dim == -1) {
          dim = src->dims().size() - 1;
        }
        int64_t start;
        if (c10::get_if<ArgNone>(&inputValues[3])) {
          start = 0;
        } else {
          start = *c10::get_if<int64_t>(&inputValues[3]);
        }
        auto start_expr = LongImm::make(start);

        int64_t end;
        ExprHandle end_expr;
        if (c10::get_if<ArgNone>(&inputValues[4])) {
          end = -1;
          end_expr = outputShape[dim];
        } else {
          end = *c10::get_if<int64_t>(&inputValues[4]);
          end_expr = LongImm::make(end);
        }

        int64_t step = *c10::get_if<int64_t>(&inputValues[5]);
        auto step_expr = LongImm::make(step);

        std::vector<ExprHandle> slice_setter_axes;
        for (int i = 0; i < axes.size(); i++) {
          slice_setter_axes.push_back(axes[i]);
        }

        slice_setter_axes[dim] = (axes[dim] - start_expr) / step_expr;

        auto cond_0 = CompareSelect::make(axes[dim], end_expr, src->load(axes),
                                          slice_setter->load(slice_setter_axes),
                                          CompareSelectOperation::kGE);

        auto cond_1 = CompareSelect::make(
            LongImm::make(0), (axes[dim] - start_expr) % step_expr,
            src->load(axes), cond_0, CompareSelectOperation::kNE);

        return cond_1;
      });
}

std::vector<ExprHandle> computePointwiseShape(
    std::vector<ArgValue> input_args) {
  return c10::get_if<BufHandle>(&input_args[0])->dims();
}

std::vector<ExprHandle> computeSelectShape(std::vector<ArgValue> input_args) {
  // TODO: DIM Must be a constant
  auto src = c10::get_if<BufHandle>(&input_args[0]);
  auto dim = *c10::get_if<int64_t>(&input_args[1]);
  // if (!dim) {
  //   std::cout << "[ERROR] Must be a constant by now!" << std::endl;
  //   throw unsupported_dtype("[ERROR] Must be a constant by now!");
  // }
  dim = dim == -1 ? src->dims().size() - 1 : dim;

  auto result = src->dims();
  result.erase(result.begin() + dim);
  return result;
}

std::vector<ExprHandle> computeSliceShape(std::vector<ArgValue> input_args) {
  auto src = c10::get_if<BufHandle>(&input_args[0]);
  auto dim = *c10::get_if<int64_t>(&input_args[1]);
  dim = dim == -1 ? src->dims().size() - 1 : dim;
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
  result[dim] =
      Min::make((end_expr - start_expr) / step_expr, src->dim(dim), true);
  return result;
}

std::vector<ExprHandle> computePermuteShape(std::vector<ArgValue> input_args) {
  auto src = c10::get_if<BufHandle>(&input_args[0]);
  auto new_index = *c10::get_if<IntList>(&input_args[1]);
  auto src_dims = src->dims();
  std::vector<ExprHandle> result;
  for (auto idx : new_index) {
    result.push_back(src_dims[idx]);
  }

  return result;
}

std::vector<ExprHandle> computeReshapeShape(std::vector<ArgValue> input_args) {
  auto src = c10::get_if<BufHandle>(&input_args[0]);
  auto shape = *c10::get_if<IntList>(&input_args[1]);
  std::vector<ExprHandle> result;

  for (auto dim : shape) {
    result.push_back(LongImm::make(dim));
  }

  auto base = LongImm::make(1);
  for (auto dim : src->dims()) {
    base = base * dim;
  }

  auto result_base = LongImm::make(1);
  for (int i = 0; i < result.size(); i++) {
    auto dim = result[i];
    if (dim.AsNode<LongImm>()->value() != -1) {
      result_base = ExprHandle(dim) * result_base;
    }
  }

  for (int i = 0; i < result.size(); i++) {
    auto dim = result[i];
    if (dim.AsNode<LongImm>()->value() == -1) {
      result[i] = base / result_base;
    }
  }

  return result;
}

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch
