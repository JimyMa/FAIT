#include "fuser/nnc_func.h"
#include "te_utils.h"
#include "tssa_set_ops.h"

namespace torch {
namespace jit {
namespace tensorexpr {

static ShapeVec computeBcastShape(SHAPE_FUNC_PARAMS) {
  auto lShape = GET_BUF_AT(0).dims(), rShape = GET_BUF_AT(1).dims();
  int64_t lRank = lShape.size(), rRank = rShape.size();
  auto outRank = std::max(lRank, rRank);
  ShapeVec outShape(outRank, int64_t(0));
  for (auto i : c10::irange(outRank)) {
    auto lIdx = lRank - 1 - i, rIdx = rRank - 1 - i;
    ExprHandle outDim;
    if (lIdx < 0)
      outDim = rShape.at(rIdx);
    else if (rIdx < 0)
      outDim = lShape.at(lIdx);
    else
      outDim = Max::make(lShape.at(lIdx), rShape.at(rIdx), true);
    outShape[outRank - 1 - i] = outDim;
  }
  return outShape;
}

static ShapeVec computeSelectShape(SHAPE_FUNC_PARAMS) {
  auto src = GET_BUF_AT(0);
  auto rank = src.dims().size();
  auto dim = GET_INT_CONST_AT(1);
  if (dim < 0) dim += rank;
  auto result = src.dims();
  result.erase(result.begin() + dim);
  return result;
}

static ShapeVec computeSliceShape(SHAPE_FUNC_PARAMS) {
  // Tensor
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
  start = IfThenElse::make(start >= int64_t(0), Min::make(start, dimSize, true),
                           start + dimSize);

  // End
  auto endVal = node->input(3);
  ExprHandle end;
  if (endVal->type()->kind() == TypeKind::NoneType)
    end = dimSize;
  else
    end = getScalarExpr<int64_t>(endVal, valueToExpr);
  end = IfThenElse::make(end >= int64_t(0), Min::make(end, dimSize, true),
                         end + dimSize);

  // Step
  int64_t step = GET_INT_CONST_AT(4);

  // Shape
  auto result = src.dims();
  result[dim] = (end - start) / step;

  return result;
}

static ShapeVec computePermuteShape(SHAPE_FUNC_PARAMS) {
  auto src = GET_BUF_AT(0);
  auto new_index = *constant_as<IntList>(node->input(1));
  auto src_dims = src.dims();
  std::vector<ExprHandle> result;
  for (auto idx : new_index) {
    result.push_back(src_dims[idx]);
  }
  return result;
}

static ShapeVec computeReshapeShape(SHAPE_FUNC_PARAMS) {
  // Count elements in source tensor
  auto src = GET_BUF_AT(0);
  auto srcShape = src.dims();
  auto srcCount =
      std::accumulate(srcShape.begin(), srcShape.end(), LongImm::make(1),
                      std::mem_fn(&ExprHandle::operator*));

  // Count elements in new tensor
  auto result = getExprList<int64_t>(node->input(1), valueToExpr);
  auto resultCount = LongImm::make(1);
  for (auto i : c10::irange(result.size())) {
    auto dim = result[i];
    auto imm = dim.AsNode<LongImm>();
    if (!imm || imm->value() != -1) resultCount = resultCount * dim;
  }

  // Fix negative dimension
  for (auto i : c10::irange(result.size())) {
    auto imm = result[i].AsNode<LongImm>();
    if (imm && imm->value() == -1) {
      result[i] = srcCount / resultCount;
      break;
    }
  }

  return result;
}

static auto _tssaSetOps = registerTssaSetOps();

OperatorMap<NNCShapeFunction> shapeFuncs{
    {"aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
     computeBcastShape},
    {"aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)",
     computeSelectShape},
    {"aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, "
     "SymInt? end=None, SymInt step=1) -> Tensor(a)",
     computeSliceShape},
    {"aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)",
     computePermuteShape},
    {"aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)",
     computeReshapeShape},
};

OperatorSet identicalShapeOps{
    "aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool "
    "non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> "
    "Tensor(a)",
    "aten::sigmoid(Tensor self) -> Tensor",
    "aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor",
    "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::div.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::eq.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::ne.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::gt.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::ge.Scalar(Tensor self, Scalar other) -> Tensor",
    "tssa::SelectSet(Tensor self, Tensor src, int dim, int index) -> Tensor",
    "tssa::SliceSet(Tensor self, Tensor src, int dim=0, SymInt? start=None, "
    "SymInt? end=None, SymInt step=1) -> Tensor",
    "tssa::Assign(Tensor self, Tensor src) -> Tensor",
};

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch
