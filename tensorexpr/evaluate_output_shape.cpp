#include "tensorexpr/evaluate_output_shape.h"

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/tensorexpr/ir.h>

using namespace torch::jit::tensorexpr;

void OutputShapeEvaluator::visit(VarPtr v) {
  if (v->name_hint() == "blockIdx.y")
    _tmp_value = degree_;
  else
    _tmp_value = dims_map_[v];
}

void OutputShapeEvaluator::visit(LongImmPtr v) { _tmp_value = v->value(); }

void OutputShapeEvaluator::visit(AddPtr v) {
  v->lhs()->accept(this);
  int64_t lhs = _tmp_value;
  v->rhs()->accept(this);
  int64_t rhs = _tmp_value;
  _tmp_value = lhs + rhs;
}

void OutputShapeEvaluator::visit(SubPtr v) {
  v->lhs()->accept(this);
  int64_t lhs = _tmp_value;
  v->rhs()->accept(this);
  int64_t rhs = _tmp_value;
  _tmp_value = lhs - rhs;
}

void OutputShapeEvaluator::visit(MulPtr v) {
  v->lhs()->accept(this);
  int64_t lhs = _tmp_value;
  v->rhs()->accept(this);
  int64_t rhs = _tmp_value;
  _tmp_value = lhs * rhs;
}

void OutputShapeEvaluator::visit(DivPtr v) {
  v->lhs()->accept(this);
  int64_t lhs = _tmp_value;
  v->rhs()->accept(this);
  int64_t rhs = _tmp_value;
  _tmp_value = lhs / rhs;
}

void OutputShapeEvaluator::visit(ModPtr v) {
  v->lhs()->accept(this);
  int64_t lhs = _tmp_value;
  v->rhs()->accept(this);
  int64_t rhs = _tmp_value;
  _tmp_value = lhs % rhs;
}

void OutputShapeEvaluator::visit(MaxPtr v) {
  v->lhs()->accept(this);
  int64_t lhs = _tmp_value;
  v->rhs()->accept(this);
  int64_t rhs = _tmp_value;
  _tmp_value = lhs > rhs ? lhs : rhs;
}

void OutputShapeEvaluator::visit(MinPtr v) {
  v->lhs()->accept(this);
  int64_t lhs = _tmp_value;
  v->rhs()->accept(this);
  int64_t rhs = _tmp_value;
  _tmp_value = lhs < rhs ? lhs : rhs;
}

void OutputShapeEvaluator::visit(CompareSelectPtr v) {
  auto options = v->compare_select_op();
  v->lhs()->accept(this);
  int64_t lhs = _tmp_value;

  v->rhs()->accept(this);
  int64_t rhs = _tmp_value;

  v->ret_val1()->accept(this);
  int64_t ret_val1 = _tmp_value;

  v->ret_val2()->accept(this);
  int64_t ret_val2 = _tmp_value;
  if (options == CompareSelectOperation::kEQ)
    _tmp_value = lhs == rhs ? ret_val1 : ret_val2;
  else if (options == CompareSelectOperation::kNE)
    _tmp_value = lhs != rhs ? ret_val1 : ret_val2;
  else if (options == CompareSelectOperation::kGE)
    _tmp_value = lhs >= rhs ? ret_val1 : ret_val2;
  else if (options == CompareSelectOperation::kGT)
    _tmp_value = lhs > rhs ? ret_val1 : ret_val2;
  else if (options == CompareSelectOperation::kLE)
    _tmp_value = lhs <= rhs ? ret_val1 : ret_val2;
  else if (options == CompareSelectOperation::kLT)
    _tmp_value = lhs < rhs ? ret_val1 : ret_val2;
}
