//
// Created by jimyma on 1/31/23.
//

#include "tensorexpr/functor_parallization.h"
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>

using namespace torch::jit::tensorexpr;

ExprPtr ParallelFunctorInputArgsMutator::expr_replace(LoadPtr functor_arg, std::vector<BufHandle>& parallel_args) {
  if (parallel_args.empty())
    return nullptr;
  if (parallel_args.size() == 1){
    LoadPtr load_op =  static_to<Load>(Load::make(parallel_args[0], ExprVectorToExprHandleVector(functor_arg->indices())).node());
    load_op->buf()->set_dims(ExprHandleVectorToExprVector(parallel_args[0].dims()));
    return load_op;
  }
  LoadPtr load_op_0 = static_to<Load>(Load::make(parallel_args[0], ExprVectorToExprHandleVector(functor_arg->indices())).node());
  LoadPtr load_op_1 = static_to<Load>(Load::make(parallel_args[1], ExprVectorToExprHandleVector(functor_arg->indices())).node());
  load_op_0->buf()->set_dims(ExprHandleVectorToExprVector(parallel_args[0].dims()));
  load_op_1->buf()->set_dims(ExprHandleVectorToExprVector(parallel_args[1].dims()));
  ExprHandle select_op = CompareSelect::make(
          ExprHandle(iter_var_),
          LongImm::make(0),
          ExprHandle(load_op_0),
          ExprHandle(load_op_1),
          CompareSelectOperation::kEQ);
  for (int i = 2; i < parallel_args.size(); i++) {
    LoadPtr load_op_i = static_to<Load>(Load::make(parallel_args[i], ExprVectorToExprHandleVector(functor_arg->indices())).node());
    load_op_i->buf()->set_dims(ExprHandleVectorToExprVector(parallel_args[i].dims()));
    select_op = CompareSelect::make(
            ExprHandle(iter_var_),
            LongImm::make(i),
            ExprHandle(load_op_i),
            select_op,
            CompareSelectOperation::kEQ);
  }
  return select_op.node();
}

ExprPtr ParallelFunctorInputArgsMutator::mutate(torch::jit::tensorexpr::LoadPtr v) {
  if (buf_args_map_.count(v->buf())) {
    auto parallel_args = buf_args_map_[v->buf()];
    return expr_replace(v, parallel_args);
  }
  return v;
}

ExprPtr ParallelFunctorInputArgsMutator::mutate(torch::jit::tensorexpr::VarPtr v) {
  if (var_args_map_.count(v)) {
    auto parallel_args = var_args_map_[v];
    if (parallel_args.empty())
      return nullptr;
    if (parallel_args.size() == 1){
      auto load_op =  var_args_map_[v][0].node();
      return load_op;
    }
    auto load_op_0 = var_args_map_[v][0].node();
    auto load_op_1 = var_args_map_[v][1].node();
    ExprHandle select_op = CompareSelect::make(
            ExprHandle(iter_var_),
            LongImm::make(0),
            ExprHandle(load_op_0),
            ExprHandle(load_op_1),
            CompareSelectOperation::kEQ);
    for (int i = 2; i < parallel_args.size(); i++) {
      auto load_op_i = var_args_map_[v][i].node();
      select_op = CompareSelect::make(
              ExprHandle(iter_var_),
              LongImm::make(i),
              ExprHandle(load_op_i),
              select_op,
              CompareSelectOperation::kEQ);
    }
    return select_op.node();
  }
  return v;
}

StmtPtr FunctorParallization::parallel_functor_load(torch::jit::tensorexpr::StmtPtr s,
                                                    int64_t degree,
                                                    VarPtr iter_var,
                                                    std::unordered_map<BufPtr, std::vector<BufHandle>> buf_args_map,
                                                    std::unordered_map<VarPtr, std::vector<VarHandle>> var_args_map) {
  ParallelFunctorInputArgsMutator mutator(degree, iter_var, buf_args_map, var_args_map);
  s->accept_mutator(&mutator);
  return s;
}

StmtPtr ParallelFunctorOutputArgsMutator::stmt_replace(StorePtr functor_arg, std::vector<BufHandle> &parallel_args) {
  if (parallel_args.empty())
    return nullptr;
  if (parallel_args.size() == 1){
    StorePtr store_op = Store::make(parallel_args[0], ExprVectorToExprHandleVector(functor_arg->indices()), ExprHandle(functor_arg->value()));
    store_op->buf()->set_dims(ExprHandleVectorToExprVector(parallel_args[0].dims()));
    return store_op;
  }
  StorePtr store_op_0 = Store::make(parallel_args[0], ExprVectorToExprHandleVector(functor_arg->indices()), ExprHandle(functor_arg->value()));
  StorePtr store_op_1 = Store::make(parallel_args[1], ExprVectorToExprHandleVector(functor_arg->indices()), ExprHandle(functor_arg->value()));
  store_op_0->buf()->set_dims(ExprHandleVectorToExprVector(parallel_args[0].dims()));
  store_op_1->buf()->set_dims(ExprHandleVectorToExprVector(parallel_args[1].dims()));
  
  StmtPtr cond = Cond::make(
          CompareSelect::make(ExprHandle(iter_var_), LongImm::make(0), CompareSelectOperation::kEQ),
          store_op_0,
          store_op_1);
  for (int i = 2; i < parallel_args.size(); i++) {
    StorePtr store_op_i = Store::make(parallel_args[i], ExprVectorToExprHandleVector(functor_arg->indices()), ExprHandle(functor_arg->value()));
    store_op_i->buf()->set_dims(ExprHandleVectorToExprVector(parallel_args[i].dims()));
    cond = Cond::make(
             CompareSelect::make(ExprHandle(iter_var_), LongImm::make(i), CompareSelectOperation::kEQ),
             store_op_i,
             cond);
  }
  return cond;
}

StmtPtr ParallelFunctorOutputArgsMutator::mutate(StorePtr v) {
  if (args_map_.count(v->buf())) {
    auto parallel_args = args_map_[v->buf()];
    return stmt_replace(v, parallel_args);
  }
  return v;
}

StmtPtr FunctorParallization::parallel_functor_store(StmtPtr s,
                                                     int64_t degree,
                                                     VarPtr iter_var,
                                                     std::unordered_map<BufPtr, std::vector<BufHandle>> args_map) {
  ParallelFunctorOutputArgsMutator mutator(degree, iter_var, args_map);
  s->accept_mutator(&mutator);
  return s;
}

ExprPtr ParallelFunctorShapeDimsMutator::expr_replace(VarPtr functor_arg, std::vector<VarHandle> &parallel_args) {
  if (parallel_args.empty())
    return nullptr;
  if (parallel_args.size() == 1)
    return parallel_args[0].node();
  ExprHandle select_op = CompareSelect::make(
          ExprHandle(iter_var_),
          LongImm::make(0),
          parallel_args[0],
          parallel_args[1],
          CompareSelectOperation::kEQ);
  for (int i = 2; i < parallel_args.size(); i++) {
    select_op = CompareSelect::make(
            ExprHandle(iter_var_),
            LongImm::make(i),
            parallel_args[i],
            select_op,
            CompareSelectOperation::kEQ);
  }
  return select_op.node();
}

ExprPtr ParallelFunctorShapeDimsMutator::mutate(VarPtr v) {
  if (args_map_.count(v)) {
    auto parallel_args = args_map_[v];
    return expr_replace(v, parallel_args);
  }
  return v;
}

StmtPtr FunctorParallization::parallel_functor_shape(StmtPtr s, int64_t degree, VarPtr iter_var,
                                                     std::unordered_map<VarPtr, std::vector<VarHandle>> args_map) {
  ParallelFunctorShapeDimsMutator mutator(degree, iter_var, args_map);
  s->accept_mutator(&mutator);
  return s;
}

