//
// Created by jimyma on 1/27/23.
//
#include <utility>

#include <c10/util/variant.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/TensorGeometry.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/util/irange.h>
#include <c10/util/string_utils.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/graph_opt.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/loopnest_randomization.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>

#include "passes/te_op.h"
#include "fuser/graph_builder.h"
#include "tensorexpr/functor_parallization.h"

using namespace torch::jit::tensorexpr;

namespace torch {
namespace jit {

GraphBuilder::GraphBuilder(const Node* node,
                           bool dyn_shape)
        : graph_(node->g(attr::Subgraph)),
          refined_types_(node->tys(c10::tssa::input_refine_types)),
          is_parallelled_args_(node->is(c10::tssa::is_parallelled_args)) {
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
  auto block = alloc<torch::jit::tensorexpr::Block>(std::vector<StmtPtr>({}));

  // TODO: Get Shape VarHandle
  auto N = LongImm::make(1);
  auto C = LongImm::make(255);
  auto H = VarHandle("H", kLong);
  auto W = VarHandle("W", kLong);

  for (int i = 0; i < graph_->inputs().size(); i++) {
    auto is_parallelled = true;
    Value* input_ = graph_->inputs()[i];

    BufHandle input_buffer(input_->debugName(), {N, C, H, W}, kDouble);
    BufferArgs_.emplace_back(input_buffer);

    bufs_[input_] = input_buffer.node();
  }

  // Step 2: Bind Node to Compute Op
  std::vector<ArgValue> inputs_expr;
  for (auto node : graph_->nodes()) {
    for (auto input_ : node->inputs()) {
      inputs_expr.emplace_back(BufHandle(bufs_[input_]));
    }
    auto output_tensor = custom_lowerings_[c10::tssa::Assign]({},
                                                              {},
                                                              {},
                                                              ScalarType::Float,
                                                              at::kCUDA);

    if (output_tensor.buf())
      bufs_[node->output(0)] = output_tensor.buf();
    block->append_stmt(output_tensor.stmt());
  }

  // Step 3: Register Output
  for (auto output : graph_->outputs()) {
    bufOutputs_.insert(bufs_[output]);
  }
  // Step 4: Functor Parallelization
  // CodeGen
  LoopNest l(block, bufOutputs_);
  LoopNest::sanitizeNames(l.root_stmt());

  if (verbose_) {
    std::cout << "Original Functor: " << std::endl;
    std::cout << to_string(l.root_stmt()) << std::endl;
  }

  l.simplify();
  l.inlineIntermediateBufs(true);

  auto stmt_ = l.root_stmt();
  if (verbose_) {
    std::cout << "after compute inline: " << std::endl;
    std::cout << to_string(stmt_) << std::endl;
  }

  // Step 2.2: Loop Binding
  for (auto buf : bufOutputs_) {
    std::vector<ForPtr> loops = l.getLoopStmtsFor(buf);
    if (loops.empty()) {
      // This happens when Buf is 0-dim
      continue;
    }
    ForPtr flattened = nullptr;
    LoopNest::flatten(loops, &flattened);
    assert(flattened);

    int loopLevels = -1;
    const int kDefaultLoopLevels = 2;

    loopLevels = (loopLevels > 0) ? loopLevels : kDefaultLoopLevels;

    int blockCount = -1;
    int blockSize = -1;

    ForPtr inner;
    const int kDefaultBlockSize = 512;
    blockSize = (blockSize > 0) ? blockSize : kDefaultBlockSize;
    LoopNest::splitWithMask(flattened, blockSize, &inner);
    flattened->set_gpu_block_index(0);
    inner->set_gpu_thread_index(0);
  }

  auto new_loop_axis = VarHandle("new_axis_i", kLong);
  stmt_ = alloc<For>(new_loop_axis.node(),
                     LongImm::make(0).node(),
                     LongImm::make(degree_).node(),
                     stmt_);
  static_to<For>(stmt_)->set_gpu_block_index(1);

  if (verbose_) {
    std::cout << "after loop binding: " << std::endl;
    std::cout << to_string(stmt_) << std::endl;
  }

  // Step 3.2: Arguments Replacement
  for (int i = 0; i < FunctorInputBufferArgs_.size(); i++) {
    if (is_parallelled_args_[i]) {
      for (int parallel_idx = 0; i < degree_; i++) {
        BufHandle parallel_buf(&"parallel_" [ parallel_idx],
                               {},
                               kFloat);
        ParallelBufferArgs_.emplace_back(parallel_buf);
      }
    } else {
        ParallelBufferArgs_.emplace_back(FunctorInputBufferArgs_[i]);
    }
  }

  for (int i = 0; i < FunctorInputShapeArgs_.size(); i++) {
    if (FunctorInputShapeArgs_[i].var()->isConstant())
      ParallelBufferArgs_.emplace_back(FunctorInputShapeArgs_[i]);
    else {
      for (int parallel_idx = 0; i < degree_; i++) {
        VarHandle parallel_shape_var("ToName", kLong);
        ParallelBufferArgs_.emplace_back(parallel_shape_var);
      }
    }
  }

  for (int i = 0; i < FunctorOutputBufferReturns_.size(); i++) {
    for (int parallel_idx = 0; i < degree_; i++) {
      BufHandle parallel_buf(&"parallel_" [ parallel_idx],
                             {},
                             kFloat);
      ParallelBufferArgs_.emplace_back(parallel_buf);
    }
  }

  stmt_ = FunctorParallization::parallel_functor_load(stmt_,
                                                      degree_,
                                                      new_loop_axis.node(),
                                                      LoadBufParallelFunctorMap,
                                                      LoadVarParallelFunctorMap);

  stmt_ = FunctorParallization::parallel_functor_store(stmt_,
                                                       degree_,
                                                       new_loop_axis.node(),
                                                       StoreBufParallelFunctorMap);

  stmt_ = FunctorParallization::parallel_functor_shape(stmt_,
                                                       degree_,
                                                       new_loop_axis.node(),
                                                       ShapeVarParallelFunctorMap);

  l.prepareForCodegen();
  l.simplify();

  auto stmt = l.root_stmt();
  IRSimplifier::simplify(stmt);

  if (verbose_) {
    std::cout << "after loop parallization: " << std::endl;
    std::cout << to_string(stmt_) << std::endl;
  }

  codegen_ = CreateCodeGen(
          "cuda_codegen",
          stmt_,
          ParallelBufferArgs_,
          device_);

}

void GraphBuilder::run(torch::jit::Stack &stack) const {
  try {
    runKernel(stack);
  } catch (...) {
    throw std::runtime_error("ParallelledFunctor Run Kernel Error");
  }
}

void GraphBuilder::runKernel(Stack &stack) const {
  auto inputs = last(stack, nInputs_);
  std::vector<at::Tensor> outputs;
}

}  // namespace jit
}  // namespace torch


