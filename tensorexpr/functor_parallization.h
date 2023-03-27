//
// Created by jimyma on 1/30/23.
//

#ifndef LONG_TAIL_FUNCTOR_PARALLIZATION_H
#define LONG_TAIL_FUNCTOR_PARALLIZATION_H

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>

#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_mutator.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"

using namespace torch::jit::tensorexpr;

ExprPtr parallel_expr(ExprPtr expr, std::vector<ExprHandle> parallel_expr);

class TORCH_API ParallelFunctorInputArgsMutator : public IRMutator {
 public:
  ParallelFunctorInputArgsMutator(
      int64_t degree, /* How many tensor in list? */
      VarPtr& iter_var,
      std::unordered_map<BufPtr, std::vector<BufHandle>>& buf_args_map,
      std::unordered_map<VarPtr, std::vector<VarHandle>>&
          var_args_map /* new args */)
      : degree_(degree),
        iter_var_(iter_var),
        buf_args_map_(buf_args_map),
        var_args_map_(var_args_map) {}
  ~ParallelFunctorInputArgsMutator() override = default;

  ExprPtr mutate(LoadPtr v) override;
  ExprPtr mutate(VarPtr v) override;

 private:
  ExprPtr expr_replace(LoadPtr functor_arg,
                       std::vector<BufHandle>& parallel_args);
  int64_t degree_;
  VarPtr iter_var_;
  std::unordered_map<BufPtr, std::vector<BufHandle>> buf_args_map_;
  std::unordered_map<VarPtr, std::vector<VarHandle>> var_args_map_;
};

class TORCH_API ParallelFunctorOutputArgsMutator : public IRMutator {
 public:
  ParallelFunctorOutputArgsMutator(
      int64_t degree, /* How many tensor in list? */
      VarPtr& iter_var,
      std::unordered_map<BufPtr, std::vector<BufHandle>>&
          args_map /* new args */)
      : degree_(degree), iter_var_(iter_var), args_map_(args_map) {}
  ~ParallelFunctorOutputArgsMutator() override = default;

  StmtPtr mutate(StorePtr v) override;

 private:
  StmtPtr stmt_replace(StorePtr functor_arg,
                       std::vector<BufHandle>& parallel_args);
  int64_t degree_;
  VarPtr iter_var_;
  std::unordered_map<BufPtr, std::vector<BufHandle>> args_map_;
};

class TORCH_API ParallelFunctorShapeDimsMutator : public IRMutator {
 public:
  ParallelFunctorShapeDimsMutator(
      int64_t degree, /* How many tensor in list? */
      VarPtr& iter_var,
      std::unordered_map<VarPtr, std::vector<VarHandle>>&
          args_map /* new args */)
      : degree_(degree), iter_var_(iter_var), args_map_(args_map) {}
  ~ParallelFunctorShapeDimsMutator() override = default;

  ExprPtr mutate(VarPtr v) override;

 private:
  ExprPtr expr_replace(VarPtr functor_arg,
                       std::vector<VarHandle>& parallel_args);
  int64_t degree_;
  VarPtr iter_var_;
  std::unordered_map<VarPtr, std::vector<VarHandle>> args_map_;
};

class FunctorParallizationMutator : public IRMutator {
 public:
  FunctorParallizationMutator(
      int64_t degree, int64_t idx,
      std::unordered_map<BufPtr, std::vector<BufHandle>> buf_args_map,
      std::unordered_map<VarPtr, std::vector<VarHandle>> var_args_map,
      std::unordered_map<BufPtr, std::vector<BufHandle>> buf_ret_map)
      : degree_(degree),
        idx_(idx),
        buf_args_map_(buf_args_map),
        var_args_map_(var_args_map),
        buf_ret_map_(buf_ret_map) {}

  ExprPtr mutate(VarPtr v) override;
  ExprPtr mutate(LoadPtr v) override;
  StmtPtr mutate(StorePtr v) override;

 private:
  int64_t degree_;
  int64_t idx_;
  std::unordered_map<BufPtr, std::vector<BufHandle>> buf_args_map_;
  std::unordered_map<VarPtr, std::vector<VarHandle>> var_args_map_;
  std::unordered_map<BufPtr, std::vector<BufHandle>> buf_ret_map_;
};

class TORCH_API FunctorParallization {
 public:
  static StmtPtr parallel_functor_load(
      StmtPtr s,      /* Functor Statement */
      int64_t degree, /* How many tensor in list? */
      VarPtr iter_var,
      std::unordered_map<BufPtr, std::vector<BufHandle>> buf_args_map,
      std::unordered_map<VarPtr, std::vector<VarHandle>> var_args_map = {});

  static StmtPtr parallel_functor_store(
      StmtPtr s, int64_t degree, VarPtr iter_var,
      std::unordered_map<BufPtr, std::vector<BufHandle>> args_map);

  static StmtPtr parallel_functor_shape(
      StmtPtr s, int64_t degree, VarPtr iter_var,
      std::unordered_map<VarPtr, std::vector<VarHandle>> dims_map);

  static StmtPtr Parallel_functor(
      StmtPtr s, int64_t degree, VarPtr iter_var,
      std::unordered_map<BufPtr, std::vector<BufHandle>> buf_args_map,
      std::unordered_map<VarPtr, std::vector<VarHandle>> var_args_map,
      std::unordered_map<BufPtr, std::vector<BufHandle>> buf_ret_map,
      std::unordered_map<VarPtr, std::vector<VarHandle>> dims_map);
};

class FunctorParallizationShapeMutator : public IRMutator {
 public:
  FunctorParallizationShapeMutator(
      int64_t degree, int64_t idx,
      std::unordered_map<VarPtr, std::vector<VarHandle>> dims_map)
      : degree_(degree), idx_(idx), dims_map_(dims_map) {}

  ExprPtr mutate(VarPtr v) override;

 private:
  int64_t degree_;
  int64_t idx_;
  std::unordered_map<VarPtr, std::vector<VarHandle>> dims_map_;
};

#endif  // LONG_TAIL_FUNCTOR_PARALLIZATION_H
