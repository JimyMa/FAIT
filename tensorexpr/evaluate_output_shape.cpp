#include "tensorexpr/evaluate_output_shape.h"
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>

using namespace torch::jit::tensorexpr;

void OutputShapeEvaluator::visit(VarPtr v) {
  std::cout << "?SDF" << std::endl;
  for (auto input_dim_message : dims_map_) {
    std::cout << to_string(input_dim_message.first) << ", " << input_dim_message.second << std::endl;
  }
  std::cout << "???: " << dims_map_[v] << std::endl;
  _tmp_value = dims_map_[v];
}

void OutputShapeEvaluator::visit(LongImmPtr v) {
  _tmp_value = v->value();
}

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
  _tmp_value = lhs > rhs ? lhs: rhs;
}

void OutputShapeEvaluator::visit(MinPtr v) {
  v->lhs()->accept(this);
  int64_t lhs = _tmp_value;
  v->rhs()->accept(this);
  int64_t rhs = _tmp_value;
  _tmp_value = lhs < rhs ? lhs: rhs;
}
