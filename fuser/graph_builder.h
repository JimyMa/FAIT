//
// Created by jimyma on 1/27/23.
//

#ifndef LONG_TAIL_GRAPH_BUILDER_H
#define LONG_TAIL_GRAPH_BUILDER_H
#include <memory>

#include "passes/tensor_ssa.h"
#include "fuser/tssa_nnc_func.h"

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

using namespace torch::jit::tensorexpr;

namespace torch {
namespace jit {

class GraphBuilder {
 public:
  GraphBuilder(const Node* node,
               bool dyn_shape = true);
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
  void runKernel(Stack& stack) const;

  bool verbose_ = true;

  int64_t degree_ = 1;
  std::vector<TypePtr> refined_types_;
  std::vector<int64_t> is_parallelled_args_;

  std::vector<CodeGen::BufferArg> FunctorInputBufferArgs_;
  std::vector<CodeGen::BufferArg> FunctorInputShapeArgs_;
  std::vector<CodeGen::BufferArg> FunctorOutputBufferReturns_;
  std::vector<CodeGen::BufferArg> ParallelBufferArgs_;

  std::unordered_map<BufPtr, std::vector<BufHandle>> LoadBufParallelFunctorMap;
  std::unordered_map<VarPtr, std::vector<VarHandle>> LoadVarParallelFunctorMap;
  std::unordered_map<VarPtr, std::vector<VarHandle>> ShapeVarParallelFunctorMap;
  std::unordered_map<BufPtr, std::vector<BufHandle>> StoreBufParallelFunctorMap;

  int64_t nInputs_ = 0;
  int64_t nOutputs_ = 0;
  at::Device device_ = at::kCUDA;
  std::vector<CodeGen::BufferArg> BufferArgs_;

  std::unordered_set<BufPtr> bufOutputs_;

  std::shared_ptr<Graph> graph_;

  std::unordered_map<const torch::jit::Value*, BufPtr> bufs_;
  std::unordered_map<const torch::jit::Value*, VarHandle> scalars_;

  std::unordered_map<c10::Symbol, NNCLoweringFunction>
  custom_lowerings_ = {{c10::tssa::Assign, computeAssign}};

  std::unique_ptr<CodeGen> codegen_;
};

}  // namespace jit
}  // namespace torch

#endif //LONG_TAIL_GRAPH_BUILDER_H
