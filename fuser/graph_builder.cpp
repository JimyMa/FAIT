//
// Created by jimyma on 1/27/23.
//
#include <utility>

#include "tvm/runtime/ndarray.h"

#include "fuser/graph_builder.h"

namespace torch {
namespace jit {

GraphBuilder::GraphBuilder(std::shared_ptr<Graph> subgraph,
                           bool dyn_shape)
        : graph_(subgraph),
          code_(subgraph, ""),
          dyn_shape_(dyn_shape) {
  allow_fallback_ = true;
  try {
    compile();
  } catch (...) {
    use_fallback_ = true;
  }
}

void GraphBuilder::compile() {
  nInputs_ = graph_->inputs().size();
  nOutputs_ = graph_->outputs().size();
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
  std::vector<tvm::runtime::NDArray> runArgs = prepareRunArgs(inputs, outputs);

}

std::vector<tvm::runtime::NDArray>
GraphBuilder::prepareRunArgs(const ArrayRef<IValue> &inputs, std::vector<at::Tensor> &outputs) const {
  std::vector<tvm::runtime::NDArray> runArgs;
  for (auto& input : inputs) {
    runArgs.emplace_back(tvm::runtime::NDArray::Empty({2, 2},
                                                      tvm::DataType::Float(32, 1),
                                                      {kDLCPU, 0}));
  }
  return runArgs;
}

}  // namespace jit
}  // namespace torch


