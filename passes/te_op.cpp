//
// Created by jimyma on 1/27/23.
//
#include "torch/csrc/jit/runtime/custom_operator.h"

#include "fuser/graph_builder.h"

#include "passes/te_op.h"

namespace torch {
namespace jit {

Operation CreateTeOperator(const Node* node) {
  auto graph_builder = std::make_shared<GraphBuilder>(node->g(attr::Subgraph));
  return [graph_builder](Stack& stack) -> int {
    graph_builder->run(stack);
    return 0;
  };
}

RegisterOperators TeOps({
  torch::jit::Operator(
    c10::tssa::Te,
    CreateTeOperator,
    AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
});

}  // namespace jit
}  // namespace torch



