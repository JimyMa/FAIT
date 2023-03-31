//
// Created by jimyma on 2/10/23.
//

#include <utility>

#include "fuser/nnc_func.h"
#include "lowering_utils.h"
#include "tssa_set_ops.h"
#include "util/logging.h"

namespace torch {
namespace jit {
namespace tensorexpr {

static Tensor computeTensor(CUSTOM_LOWERING_PARAMS) {
  return Compute("tensor", outShape, [&](const ParameterList& axes) {
    return GET_ANY_SCALAR_EXPR_AT(0);
  });
}

template <class T>
static CustomLoweringFunction getComputeFillConst(T val,
                                                  const std::string& name) {
  return [=](CUSTOM_LOWERING_PARAMS) {
    return Compute("fill_" + name, outShape, [&](const ParameterList& axes) {
      return ExprHandle(getImmediateByType(Dtype(outDtype), val));
    });
  };
}

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
static CustomLoweringFunction getComputeBinaryBcast(BinOp&& op,
                                                    const std::string& name) {
  return [=](CUSTOM_LOWERING_PARAMS) {
    return Compute(name + "_bcast", outShape, [&](const ParameterList& axes) {
      auto lhs = loadBcast(GET_BUF_AT(0), outShape, axes);
      auto rhs = loadBcast(GET_BUF_AT(1), outShape, axes);
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

using EwiseFunc = std::function<ExprHandle(const ExprHandle&)>;
#define EWISE_FUNC_CREATOR_PARAMS Node *node, const ValueExprMap &valueToExpr
using EwiseFuncCreator = std::function<EwiseFunc(EWISE_FUNC_CREATOR_PARAMS)>;

static EwiseFunc getClamp(EWISE_FUNC_CREATOR_PARAMS) {
  return [node, &valueToExpr](const ExprHandle& src) {
    auto result = src;
    if (node->input(1)->type()->kind() != TypeKind::NoneType)
      result = Max::make(
          result,
          ExprHandle(Cast::make(src.dtype(), GET_ANY_SCALAR_EXPR_AT(1))), true);
    if (node->input(2)->type()->kind() != TypeKind::NoneType)
      result = Min::make(
          result,
          ExprHandle(Cast::make(src.dtype(), GET_ANY_SCALAR_EXPR_AT(2))), true);
    return result;
  };
};

static OperatorMap<EwiseFuncCreator> ewiseExprCreators{
    {"aten::sigmoid(Tensor self) -> Tensor",
     [](EWISE_FUNC_CREATOR_PARAMS) { return sigmoid; }},
    {"aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor",
     getClamp},
};

static std::vector<EwiseFunc> findEwiseFuncsForSetSrc(
    Value* self, Value* src, Symbol viewSym, const ValueExprMap& valueToExpr) {
  // Trace src to self
  Value* curVal = src;
  std::list<EwiseFunc> funcList;

  while (true) {
    // Check definition of value
    auto node = curVal->node();

    // If view symbol is encountered, check if it operates on self
    if (node->kind() == viewSym) {
      if (node->input(0) == self)
        break;
      else
        return {};
    }

    // Not the target view symbol, check if we can create an element-wise
    // expression
    auto op = node->maybeOperator();
    if (!op) return {};
    auto exprCreator = ewiseExprCreators.find(*op);
    if (!exprCreator) return {};
    funcList.push_front((*exprCreator)(node, valueToExpr));

    // Move to its first input
    curVal = node->input(0);
  }

  return {funcList.begin(), funcList.end()};
}

static Tensor computeSliceSet(CUSTOM_LOWERING_PARAMS) {
  return Compute("slice_set", outShape, [&](const ParameterList& axes) {
    // Tensor
    auto self = GET_BUF_AT(0);
    auto rank = self.dims().size();
    auto src = GET_BUF_AT(1);
    auto dim = GET_INT_CONST_AT(2);
    if (dim < 0) dim += rank;
    auto dimSize = self.dims().at(dim);

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

    // Setter axes
    std::vector<ExprHandle> srcAxes(axes.begin(), axes.end());
    auto dimAxis = axes[dim];
    srcAxes[dim] = (axes[dim] - start) / step;

    // See if we can create an elementwise pipeline for source values
    auto srcElem = src.load(srcAxes);
    auto ewiseFuncs = findEwiseFuncsForSetSrc(node->input(0), node->input(1),
                                              aten::slice, valueToExpr);
    if (!ewiseFuncs.empty()) {
      srcElem = self.load(axes);
      for (auto& func : ewiseFuncs) srcElem = func(srcElem);
    }

    // Select elements
    auto notSet = (dimAxis < start) || (dimAxis >= end) ||
                  ((dimAxis - start) % step != int64_t(0));
    auto result = IfThenElse::make(notSet, self.load(axes), srcElem);

    return result;
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

static Tensor computeCat(CUSTOM_LOWERING_PARAMS) {
  return Compute("cat", outShape, [&](const ParameterList& axes) {
    // Process dimension
    auto bufs = GET_BUF_LIST_AT(0);
    auto inRank = bufs.front().dims().size();
    auto dim = GET_INT_CONST_AT(1);
    if (dim < 0) dim += inRank;

    // Compute section index range
    std::vector<ExprHandle> indices(bufs.size() + 1);
    indices[0] = int64_t(0);
    for (auto i : c10::irange(bufs.size()))
      indices[i + 1] = indices[i] + bufs[i].dim(dim);

    // Switch buffers according to index range at concatenation axis
    auto dimAxis = axes[dim];
    std::vector<ExprHandle> loadAxes(axes.begin(), axes.end());
    ExprHandle result(getImmediateByType(bufs.front().dtype(), 0));
    for (int64_t i = bufs.size() - 1; i >= 0; i--) {
      auto bufLoadAxes = loadAxes;
      bufLoadAxes[dim] = dimAxis - indices[i];
      result = IfThenElse::make(ExprHandle(dimAxis) < indices[i + 1],
                                bufs[i].load(bufLoadAxes), result);
    }

    return result;
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
    {"aten::tensor.int(int t, *, ScalarType? dtype=None, Device? device=None, "
     "bool requires_grad=False) -> Tensor",
     computeTensor},
    {"aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? "
     "layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
     getComputeFillConst(0, "zeros")},
    {"aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, "
     "Layout? layout=None, Device? device=None, bool? pin_memory=None) -> "
     "Tensor",
     computeArange},
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
    {"aten::cat(Tensor[] tensors, int dim=0) -> Tensor", computeCat},
    {"aten::stack(Tensor[] tensors, int dim=0) -> Tensor", computeStack},
};

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch
