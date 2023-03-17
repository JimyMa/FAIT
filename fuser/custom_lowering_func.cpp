//
// Created by jimyma on 2/10/23.
//

#include <utility>

#include "fuser/nnc_func.h"
#include "te_utils.h"
#include "tssa_set_ops.h"
#include "util/logging.h"

namespace torch {
namespace jit {
namespace tensorexpr {

static Tensor computeArange(CUSTOM_LOWERING_PARAMS) {
  return Compute("arange", outShape, [&](const VarHandle& i) {
    auto start = GET_INT_EXPR_AT(0);
    return Cast::make(Dtype(outDtype), start + i);
  });
}

static ExprHandle loadBcast(const BufHandle& srcBuf, const ShapeVec& dstShape,
                            const ParameterList& axes) {
  auto srcRank = srcBuf.dims().size(), dstRank = dstShape.size();
  std::vector<ExprHandle> loadAxes(axes.begin(), axes.end());
  loadAxes.erase(loadAxes.begin(), loadAxes.begin() + dstRank - srcRank);
  for (auto i : c10::irange(srcRank)) {
    loadAxes[i] =
        IfThenElse::make(srcBuf.dim(i) == int64_t(1), int64_t(0), loadAxes[i]);
  }
  return srcBuf.load(loadAxes);
}

template <class BinOp>
static CustomLoweringFunction getComputeBinaryBcast(BinOp&& op) {
  return [&](CUSTOM_LOWERING_PARAMS) {
    return Compute("binary_bcast", outShape, [&](const ParameterList& axes) {
      auto lhs = loadBcast(GET_BUF_AT(0), outShape, axes);
      auto rhs = loadBcast(GET_BUF_AT(1), outShape, axes);
      return op(lhs, rhs);
    });
  };
}

template <class BinOp>
static CustomLoweringFunction getComputeBinaryScalar(BinOp&& op) {
  return [op](CUSTOM_LOWERING_PARAMS) {
    return Compute("binary_scalar", outShape, [&](const ParameterList& axes) {
      auto lhs = GET_BUF_AT(0).load(axes);
      auto rhs = Cast::make(Dtype(outDtype), GET_ANY_SCALAR_EXPR_AT(1));
      return op(lhs, rhs);
    });
  };
}

static Tensor computeSelect(CUSTOM_LOWERING_PARAMS) {
  return Compute("select", outShape, [&](const ParameterList& axes) {
    auto src = GET_BUF_AT(0);
    auto rank = src.dims().size();
    auto dim = GET_INT_CONST_AT(1);
    if (dim < 0) dim += rank;
    auto dimSize = src.dims().at(dim);
    auto idx = GET_INT_EXPR_AT(2);
    idx = IfThenElse::make(idx >= int64_t(0), idx, idx + dimSize);

    std::vector<ExprHandle> output_idx(axes.begin(), axes.end());
    output_idx.insert(output_idx.begin() + dim, idx);

    return src.load(output_idx);
  });
}

static Tensor computeSelectSet(CUSTOM_LOWERING_PARAMS) {
  return Compute("select_set", outShape, [&](const ParameterList& axes) {
    auto src = GET_BUF_AT(0);
    auto rank = src.dims().size();
    auto select_setter = GET_BUF_AT(1);
    auto dim = GET_INT_CONST_AT(2);
    if (dim < 0) dim += rank;
    auto dimSize = src.dims().at(dim);
    auto idx = GET_INT_EXPR_AT(3);
    idx = IfThenElse::make(idx >= int64_t(0), idx, idx + dimSize);

    std::vector<ExprHandle> setter_idx(axes.begin(), axes.end());
    setter_idx.erase(setter_idx.begin() + dim);
    auto cond =
        CompareSelect::make(axes[dim], idx, select_setter.load(setter_idx),
                            src.load(axes), CompareSelectOperation::kEQ);

    return cond;
  });
}

static Tensor computeSlice(CUSTOM_LOWERING_PARAMS) {
  return Compute("slice", outShape, [&](const ParameterList& axes) {
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
  return Compute("slice_set", outShape, [&](const ParameterList& axes) {
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
    start = IfThenElse::make(start >= int64_t(0),
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
    auto cond_1 = CompareSelect::make(axes[dim], start, src.load(axes), cond_0,
                                      CompareSelectOperation::kLT);
    auto cond_2 = CompareSelect::make(int64_t(0), (axes[dim] - start) % step,
                                      src.load(axes), cond_1,
                                      CompareSelectOperation::kNE);

    return cond_2;
  });
}

static Tensor computeAssign(CUSTOM_LOWERING_PARAMS) {
  return Compute("assign", outShape, [&](const ParameterList& axes) {
    // Tensor
    auto src = GET_BUF_AT(0);
    auto assigner = GET_BUF_AT(1);
    return assigner;
  });
}

static Tensor computeUnsqueeze(CUSTOM_LOWERING_PARAMS) {
  return Compute("unsqueeze", outShape, [&](const ParameterList& axes) {
    auto self = GET_BUF_AT(0);
    auto rank = self.dims().size();
    auto dim = GET_INT_CONST_AT(1);
    if (dim < 0) dim += rank + 1;
    auto loadAxes = axes;
    loadAxes.erase(loadAxes.begin() + dim);
    return self.load(loadAxes);
  });
}

static Tensor computeRepeat(CUSTOM_LOWERING_PARAMS) {
  return Compute("repeat", outShape, [&](const ParameterList& axes) {
    // Remove front axes
    auto self = GET_BUF_AT(0);
    auto inShape = self.dims();
    auto inRank = inShape.size(), outRank = outShape.size();
    std::vector<ExprHandle> loadAxes(axes.begin(), axes.end());
    loadAxes.erase(loadAxes.begin(), loadAxes.begin() + outRank - inRank);

    // Update load axes
    for (auto i : c10::irange(inRank)) {
      const auto& axis = loadAxes[i];
      loadAxes[i] =
          IfThenElse::make(inShape[i] == outShape[i], axis, axis % inShape[i]);
    }

    return self.load(loadAxes);
  });
}

static Tensor computeStack(CUSTOM_LOWERING_PARAMS) {
  return Compute("stack", outShape, [&](const ParameterList& axes) {
    // Process dimension
    auto bufs = GET_BUF_LIST_AT(0);
    auto inRank = bufs.front().dims().size();
    auto dim = GET_INT_CONST_AT(1);
    if (dim < 0) dim += inRank + 1;

    // Switch buffers according to dim axis
    auto dimAxis = axes[dim];
    std::vector<ExprHandle> loadAxes(axes.begin(), axes.end());
    loadAxes.erase(loadAxes.begin() + dim);
    ExprHandle result(getImmediateByType(bufs.front().dtype(), 0));
    for (int64_t i = bufs.size() - 1; i >= 0; i--) {
      result = IfThenElse::make(ExprHandle(dimAxis) == i,
                                bufs[i].load(loadAxes), result);
    }

    return result;
  });
}

static auto _tssaSetOps = registerTssaSetOps();

OperatorMap<CustomLoweringFunction> customLoweringFuncs{
    {"aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, "
     "Layout? layout=None, Device? device=None, bool? pin_memory=None) -> "
     "Tensor",
     computeArange},
    // {"aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    // getComputeBinaryScalar(&Add::make)},
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
    {"aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)", computeUnsqueeze},
    {"aten::repeat(Tensor self, SymInt[] repeats) -> Tensor", computeRepeat},
    {"aten::stack(Tensor[] tensors, int dim=0) -> Tensor", computeStack},
    {"tssa::Assign(Tensor self, Tensor src) -> Tensor", computeAssign},
};

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch
