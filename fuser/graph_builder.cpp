//
// Created by jimyma on 1/27/23.
//
#include <utility>

#include "passes/te_op.h"
#include "fuser/graph_builder.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
//#include "torch/csrc/jit/tensorexpr/types.h"

using namespace torch::jit::tensorexpr;

namespace torch {
namespace jit {

GraphBuilder::GraphBuilder(const Node* node,
                           bool dyn_shape)
        : graph_(node->g(attr::Subgraph)),
          code_(node->g(attr::Subgraph), ""),
          refined_types_(node->tys(c10::tssa::input_refine_types)),
          is_parallelled_args_(node->is(c10::tssa::is_parallelled_args)),
          dyn_shape_(dyn_shape) {
  allow_fallback_ = true;
  try {
    compile();
  } catch (...) {
    throw std::runtime_error("Functor Parallization Compile Error!!");
  }
}

void GraphBuilder::compile() {
  nInputs_ = graph_->inputs().size();
  nOutputs_ = graph_->outputs().size();

  // Step 0: find common shape and dyn shape
  // For Common shape, use it by LongImm.
  // For Dyn shape, use it by VarHandle.

  // Step 1: Bind inputs to buffers.
//  auto block = alloc<torch::jit::tensorexpr::Block>(std::vector<StmtPtr>({}));
//
//  auto N = LongImm::make(1);
//  auto C = LongImm::make(255);
//  auto H = VarHandle("H", kLong);
//  auto W = VarHandle("W", kLong);
//
//  for (int i = 0; i < graph_->inputs().size(); i++) {
//    auto is_parallelled = true;
//    Value* input_ = graph_->inputs()[i];
//
//    BufHandle input_buffer(input_->debugName(), {N, C, H, W}, kDouble);
//    BufferArgs_.emplace_back(input_buffer);
//  }
//
//  // Step 2: Bind Node to Compute Op
//  std::vector<ExprHandle> inputs_expr;
//  for (auto node : graph_->nodes()) {
//
//  }
//
//  // Step 3: Register Output
//
//  // Step 4: Functor Parallelization

}

void GraphBuilder::run(torch::jit::Stack &stack) const {
  if (!use_fallback_ && !allow_fallback_) {
    runKernel(stack);
  } else if (!use_fallback_ && allow_fallback_) {
    try {
      runKernel(stack);
    } catch (...) {
      fallback(stack);
    }
  } else {
    fallback(stack);
  }
}

void GraphBuilder::runKernel(Stack &stack) const {
  auto inputs = last(stack, nInputs_);
  std::vector<at::Tensor> outputs;
}

}  // namespace jit
}  // namespace torch


