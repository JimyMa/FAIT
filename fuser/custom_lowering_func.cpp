//
// Created by jimyma on 2/10/23.
//

#include <utility>

#include "fuser/nnc_func.h"
#include "tssa_set_ops.h"
#include "util/logging.h"
#include "util/types.h"

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

        auto cond_1 =
            CompareSelect::make(axes[dim], start_expr, src->load(axes), cond_0,
                                CompareSelectOperation::kLT);

        auto cond_2 = CompareSelect::make(
            LongImm::make(0), (axes[dim] - start_expr) % step_expr,
            src->load(axes), cond_1, CompareSelectOperation::kNE);

        return cond_2;
      });
}

static Tensor computeSelectNew(CUSTOM_LOWERING_PARAMS) {
  return Compute("select", outShape, [&](const std::vector<VarHandle>& axes) {
    auto src = GET_BUF_AT(0);
    auto rank = src.dims().size();
    auto dim = GET_INT_CONST_AT(1);
    if (dim < 0) dim += rank;
    auto idx = GET_INT_SCALAR_EXPR_AT(2);
    idx = IfThenElse::make(idx >= 0, idx, idx + int64_t(rank));

    std::vector<ExprHandle> output_idx(axes.begin(), axes.end());
    output_idx.insert(output_idx.begin() + dim, idx);

    return src.load(output_idx);
  });
}

static Tensor computeSelectSetNew(CUSTOM_LOWERING_PARAMS) {
  return Compute(
      "select_set", outShape, [&](const std::vector<VarHandle>& axes) {
        auto src = GET_BUF_AT(0);
        auto rank = src.dims().size();
        auto select_setter = GET_BUF_AT(1);
        auto dim = GET_INT_CONST_AT(2);
        if (dim < 0) dim += rank;
        auto idx = GET_INT_SCALAR_EXPR_AT(3);
        idx = IfThenElse::make(idx >= 0, idx, idx + int64_t(rank));

        std::vector<ExprHandle> setter_idx(axes.begin(), axes.end());
        setter_idx.erase(setter_idx.begin() + dim);
        auto cond =
            CompareSelect::make(axes[dim], idx, select_setter.load(setter_idx),
                                src.load(axes), CompareSelectOperation::kEQ);

        return cond;
      });
}

static Tensor computeSliceNew(CUSTOM_LOWERING_PARAMS) {
  return Compute("slice", outShape, [&](const std::vector<VarHandle>& axes) {
    // Source tensor
    auto src = GET_BUF_AT(0);
    auto rank = src.dims().size();
    auto dim = GET_INT_CONST_AT(1);
    if (dim < 0) dim += rank;
    auto dimSize = src.dims().at(dim);
    // Start
    auto startVal = node->input(2);
    ExprHandle start;
    if (startVal->type()->kind() == TypeKind::NoneType)
      start = LongImm::make(0);
    else
      start = getScalarExpr<int64_t>(startVal, valueToExpr);
    start = IfThenElse::make(start >= LongImm::make(0), start, start + dimSize);
    // start = IfThenElse::make(start >= LongImm::make(0),
    //                          Min::make(start, dimSize, true), start +
    //                          dimSize);

    // Step
    int64_t step = GET_INT_CONST_AT(4);
    // Source indices
    std::vector<ExprHandle> output_idx(axes.begin(), axes.end());
    output_idx[dim] = start + LongImm::make(step) * output_idx[dim];

    return src.load(output_idx);
  });
}

static Tensor computeSliceSetNew(CUSTOM_LOWERING_PARAMS) {
  return Compute(
      "slice_set", outShape, [&](const std::vector<VarHandle>& axes) {
        // Tensor
        auto src = GET_BUF_AT(0);
        auto rank = src.dims().size();
        auto slice_setter = GET_BUF_AT(1);
        auto dim = GET_INT_CONST_AT(2);
        if (dim < 0) dim += rank;
        auto dimSize = src.dims().at(dim);

        // Start
        auto startVal = node->input(3);
        ExprHandle start;
        if (startVal->type()->kind() == TypeKind::NoneType)
          start = LongImm::make(0);
        else
          start = getScalarExpr<int64_t>(startVal, valueToExpr);
        start =
            IfThenElse::make(start >= int64_t(0),
                             Min::make(start, dimSize, true), start + dimSize);

        // End
        auto endVal = node->input(4);
        ExprHandle end;
        if (endVal->type()->kind() == TypeKind::NoneType)
          end = dimSize;
        else
          end = getScalarExpr<int64_t>(endVal, valueToExpr);
        end = IfThenElse::make(end >= LongImm::make(0),
                               Min::make(end, dimSize, true), end + dimSize);

        // Step
        int64_t step = GET_INT_CONST_AT(5);

        std::vector<ExprHandle> slice_setter_axes;
        for (int i = 0; i < axes.size(); i++) {
          slice_setter_axes.push_back(axes[i]);
        }

        // Index
        slice_setter_axes[dim] = (axes[dim] - start) / step;
        auto cond_0 = CompareSelect::make(axes[dim], end, src.load(axes),
                                          slice_setter.load(slice_setter_axes),
                                          CompareSelectOperation::kGE);
        auto cond_1 = CompareSelect::make(axes[dim], start, src.load(axes),
                                          cond_0, CompareSelectOperation::kLT);
        auto cond_2 = CompareSelect::make(
            LongImm::make(0), (axes[dim] - start) % step, src.load(axes),
            cond_1, CompareSelectOperation::kNE);

        return cond_2;
      });
}

static Tensor computeAssignNew(CUSTOM_LOWERING_PARAMS) {
  return Compute("assign", outShape, [&](const std::vector<VarHandle>& axes) {
    // Tensor
    auto src = GET_BUF_AT(0);
    auto assigner = GET_BUF_AT(1);
    return assigner;
  });
}

static auto _tssaSetOps = registerTssaSetOps();

OperatorMap<CustomLoweringFunction> customLoweringFuncs{
    {"aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)",
     computeSelectNew},
    {"tssa::SelectSet(Tensor self, Tensor src, int dim, int index) -> Tensor",
     computeSelectSetNew},
    {"aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, "
     "SymInt? end=None, SymInt step=1) -> Tensor(a)",
     computeSliceNew},
    {"tssa::SliceSet(Tensor self, Tensor src, int dim=0, SymInt? start=None, "
     "SymInt? end=None, SymInt step=1) -> Tensor",
     computeSliceSetNew},
    {"tssa::Assign(Tensor self, Tensor src) -> Tensor", computeAssignNew}};

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch
