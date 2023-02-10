//
// Created by jimyma on 1/27/23.
//

#ifndef LONG_TAIL_GRAPH_BUILDER_H
#define LONG_TAIL_GRAPH_BUILDER_H
#include <memory>

#include "torch/csrc/jit/ir/ir.h"
//#include "torch/csrc/jit/tensorexpr/codegen.h"
#include "torch/csrc/jit/runtime/interpreter.h"

//using namespace torch::jit::tensorexpr;

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

  void fallback(Stack& stack) const {
    InterpreterState(code_).run(stack);
  }

  void recompile();

  const std::shared_ptr<Graph> graph() {
    return graph_;
  }

 private:
  void compile();
  void runKernel(Stack& stack) const;

  int64_t nInputs_ = 0;
  int64_t nOutputs_ = 0;
  at::Device device_ = at::kCUDA;
//  std::vector<CodeGen::BufferArg> BufferArgs_;

  std::shared_ptr<Graph> graph_;

  std::vector<TypePtr> refined_types_;
  std::vector<int64_t> is_parallelled_args_;

  bool dyn_shape_;
  Code code_;
  bool allow_fallback_{false};
  bool use_fallback_{false};
  
};

}  // namespace jit
}  // namespace torch

#endif //LONG_TAIL_GRAPH_BUILDER_H
