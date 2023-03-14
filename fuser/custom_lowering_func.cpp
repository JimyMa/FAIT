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

static Tensor computeSelect(CUSTOM_LOWERING_PARAMS) {
  return Compute("select", outShape, [&](const std::vector<VarHandle>& axes) {
    auto src = GET_BUF_AT(0);
    auto rank = src.dims().size();
    auto dim = GET_INT_CONST_AT(1);
    if (dim < 0) dim += rank;
    auto idx = GET_INT_SCALAR_EXPR_AT(2);
    idx = IfThenElse::make(idx >= int64_t(0), idx, idx + int64_t(rank));

    std::vector<ExprHandle> output_idx(axes.begin(), axes.end());
    output_idx.insert(output_idx.begin() + dim, idx);

    return src.load(output_idx);
  });
}

static Tensor computeSelectSet(CUSTOM_LOWERING_PARAMS) {
  return Compute(
      "select_set", outShape, [&](const std::vector<VarHandle>& axes) {
        auto src = GET_BUF_AT(0);
        auto rank = src.dims().size();
        auto select_setter = GET_BUF_AT(1);
        auto dim = GET_INT_CONST_AT(2);
        if (dim < 0) dim += rank;
        auto idx = GET_INT_SCALAR_EXPR_AT(3);
        idx = IfThenElse::make(idx >= int64_t(0), idx, idx + int64_t(rank));

        std::vector<ExprHandle> setter_idx(axes.begin(), axes.end());
        setter_idx.erase(setter_idx.begin() + dim);
        auto cond =
            CompareSelect::make(axes[dim], idx, select_setter.load(setter_idx),
                                src.load(axes), CompareSelectOperation::kEQ);

        return cond;
      });
}

static Tensor computeSlice(CUSTOM_LOWERING_PARAMS) {
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
      start = int64_t(0);
    else
      start = getScalarExpr<int64_t>(startVal, valueToExpr);
    start = IfThenElse::make(start >= int64_t(0),
                             Min::make(start, dimSize, true), start + dimSize);

    // Step
    int64_t step = GET_INT_CONST_AT(4);
    // Source indices
    std::vector<ExprHandle> output_idx(axes.begin(), axes.end());
    output_idx[dim] = start + LongImm::make(step) * output_idx[dim];

    return src.load(output_idx);
  });
}

static Tensor computeSliceSet(CUSTOM_LOWERING_PARAMS) {
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
          start = int64_t(0);
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
        end = IfThenElse::make(end >= int64_t(0), Min::make(end, dimSize, true),
                               end + dimSize);

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
            int64_t(0), (axes[dim] - start) % step, src.load(axes), cond_1,
            CompareSelectOperation::kNE);

        return cond_2;
      });
}

static Tensor computeAssign(CUSTOM_LOWERING_PARAMS) {
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
     computeSelect},
    {"tssa::SelectSet(Tensor self, Tensor src, int dim, int index) -> Tensor",
     computeSelectSet},
    {"aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, "
     "SymInt? end=None, SymInt step=1) -> Tensor(a)",
     computeSlice},
    {"tssa::SliceSet(Tensor self, Tensor src, int dim=0, SymInt? start=None, "
     "SymInt? end=None, SymInt step=1) -> Tensor",
     computeSliceSet},
    {"tssa::Assign(Tensor self, Tensor src) -> Tensor", computeAssign}};

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch
