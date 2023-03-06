//
// Created by jimyma on 1/27/23.
//

#ifndef LONG_TAIL_GRAPH_BUILDER_H
#define LONG_TAIL_GRAPH_BUILDER_H
#include <ATen/Context.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/core/interned_strings.h>

#include <c10/cuda/CUDAFunctions.h>

#include <functional>
#include <memory>

#include "fuser/solve_update.h"
#include "passes/tensor_ssa.h"
#include "fuser/tssa_nnc_func.h"

#include <string>
// #include <torch/csrc/jit/ir/ir.h>
// #include <torch/csrc/jit/runtime/interpreter.h>
// #include <torch/csrc/jit/tensorexpr/analysis.h>
// #include <torch/csrc/jit/tensorexpr/codegen.h>
// #include <torch/csrc/jit/tensorexpr/expr.h>
// #include <torch/csrc/jit/tensorexpr/fwd_decls.h>
// #include <torch/csrc/jit/tensorexpr/lowerings.h>
// #include <torch/csrc/jit/tensorexpr/tensor.h>
// #include <torch/csrc/jit/tensorexpr/types.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/csrc/jit/tensorexpr/operators/misc.h>
#include <unordered_map>
#include <vector>

using namespace torch::jit::tensorexpr;


namespace torch {
namespace jit {

using NNCShapeFunction = std::function<std::vector<ExprHandle>(std::vector<ArgValue>)>;

class GraphBuilder {
 public:
  GraphBuilder(const Node* node);
  void run(Stack& stack) const;

  void runFast(
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs) const;

  void recompile();

  const std::shared_ptr<Graph> graph() {
    return graph_;
  }

 private:
  void compile();
  std::vector<ArgValue> get_input_expr(Node* node);
  std::vector<int64_t> get_stride_by_shape(const std::vector<int64_t> shape) const;
  std::vector<ExprHandle> get_stride_by_expr_dims(const std::vector<ExprHandle> expr_shape) const;
  Dtype get_scalar_type_by_value_type(TypePtr value_type);
  VarHandle get_const_var_by_value(const Value* value);

  std::vector<CodeGen::CallArg> prepareRunArgs(
      const at::ArrayRef<IValue>& inputs,
      std::vector<std::vector<at::Tensor>>& outputs) const;
  void runKernel(Stack& stack) const;
  std::string set_hash_name(std::string base_name) {
    if (!name_hash_map_.count(base_name)) {
      name_hash_map_[base_name] = 0;
    }
    std::string result = base_name + "_" + std::to_string(name_hash_map_[base_name]);
    name_hash_map_[base_name] += 1;
    return result;
  }

  bool verbose_ = false;

  int64_t degree_ = 1;
  std::vector<TypePtr> refined_types_;
  std::vector<int64_t> is_parallelled_args_;

  std::map<Value*, std::vector<ExprHandle>> FunctorShapeMap_;
  std::map<const torch::jit::Value*, BufPtr> bufs_;
  std::map<const torch::jit::Value*, VarPtr> vars_;

  // std::vector<CodeGen::BufferArg> FunctorInputBufferArgs_;
  // std::vector<CodeGen::BufferArg> FunctorInputShapeArgs_;
  // std::vector<CodeGen::BufferArg> FunctorOutputBufferReturns_;
  std::vector<CodeGen::BufferArg> ParallelBufferArgs_;

  std::unordered_map<std::string, int> name_hash_map_;

  std::unordered_map<BufPtr, std::vector<BufHandle>> LoadBufParallelFunctorMap;
  std::unordered_map<VarPtr, std::vector<VarHandle>> LoadVarParallelFunctorMap;
  std::unordered_map<VarPtr, std::vector<VarHandle>> ShapeVarParallelFunctorMap;
  std::unordered_map<BufPtr, std::vector<BufHandle>> StoreBufParallelFunctorMap;

  int64_t nInputs_ = 0;
  int64_t nOutputs_ = 0;
  at::Device device_ = at::Device(at::kCUDA, at::cuda::current_device());
  std::vector<CodeGen::BufferArg> BufferArgs_;

  std::unordered_set<BufPtr> bufOutputs_;

  std::shared_ptr<Graph> graph_;

  std::unordered_map<c10::Symbol, NNCLoweringFunction>
  custom_lowerings_ = {{c10::tssa::Assign, computeAssign},
                       {c10::aten::select, computeSelect},
                       {c10::aten::slice, computeSlice},
                       {c10::tssa::SliceSet, computeSliceSet},
                       {c10::tssa::SelectSet, computeSelectSet}};
  
  std::unordered_map<c10::Symbol, NNCShapeFunction>
  shape_func = {{c10::aten::add, computePointwiseShape},
                {c10::aten::select, computeSelectShape},
                {c10::aten::slice, computeSliceShape},
                {c10::aten::permute, computePermuteShape},
                {c10::aten::reshape, computeReshapeShape}};

  std::unique_ptr<CodeGen> codegen_;
};

}  // namespace jit
}  // namespace torch

#endif //LONG_TAIL_GRAPH_BUILDER_H
