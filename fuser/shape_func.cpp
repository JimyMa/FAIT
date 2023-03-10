#include "fuser/nnc_func.h"
#include "tssa_set_ops.h"

namespace torch {
namespace jit {
namespace tensorexpr {

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

static ShapeVec computeSelectShapeNew(SHAPE_FUNC_PARAMS) {
  auto src = GET_BUF_AT(0);
  auto rank = src.dims().size();
  auto dim = GET_INT_CONST_AT(1);
  if (dim < 0) dim += rank;
  auto result = src.dims();
  result.erase(result.begin() + dim);
  return result;
}

static ShapeVec computeSliceShapeNew(SHAPE_FUNC_PARAMS) {
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
  start = IfThenElse::make(start >= 0, Min::make(start, dimSize, true),
                           start + dimSize);

  // End
  auto endVal = node->input(3);
  ExprHandle end;
  if (endVal->type()->kind() == TypeKind::NoneType)
    end = dimSize;
  else
    end = getScalarExpr<int64_t>(endVal, valueToExpr);
  end =
      IfThenElse::make(end >= 0, Min::make(end, dimSize, true), end + dimSize);

  // Step
  int64_t step = GET_INT_CONST_AT(4);

  // Shape
  auto result = src.dims();
  result[dim] = (end - start) / step;

  return result;
}

static ShapeVec computePermuteShapeNew(SHAPE_FUNC_PARAMS) {
  auto src = GET_BUF_AT(0);
  auto new_index = *constant_as<IntList>(node->input(1));
  auto src_dims = src.dims();
  std::vector<ExprHandle> result;
  for (auto idx : new_index) {
    result.push_back(src_dims[idx]);
  }
  return result;
}

static ShapeVec computeReshapeShapeNew(SHAPE_FUNC_PARAMS) {
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
    {"aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)",
     computeSelectShapeNew},
    {"aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, "
     "SymInt? end=None, SymInt step=1) -> Tensor(a)",
     computeSliceShapeNew},
    {"aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)",
     computePermuteShapeNew},
    {"aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)",
     computeReshapeShapeNew},
};

OperatorSet identicalShapeOps{
    "aten::sigmoid(Tensor self) -> Tensor",
    "tssa::SelectSet(Tensor self, Tensor src, int dim, int index) -> Tensor",
    "tssa::SliceSet(Tensor self, Tensor src, int dim=0, SymInt? start=None, "
    "SymInt? end=None, SymInt step=1) -> Tensor",
};

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch
