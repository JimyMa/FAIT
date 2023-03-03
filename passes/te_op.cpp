//
// Created by jimyma on 1/27/23.
//
#include "passes/type_utils.h"
#include "torch/csrc/jit/runtime/custom_operator.h"

#include "util/ir.h"
#include "fuser/graph_builder.h"
#include "fuser/solve_update.h"
#include "passes/parallelize_loops.h"

#include "passes/te_op.h"
#include <memory>


namespace torch {
namespace jit {

Node* GetParallelledFunctorByParallelMap(Node* node,
                                         std::unordered_map<Value*, TypePtr>& refine_types) {
  // Construct an op
  auto functor_op = node->owningGraph()->createWithSubgraph(c10::tssa::ParallelledFunctor);

  auto subgraph = functor_op->g(attr::Subgraph);

  // Get input_degree by tuple lengths, not by iter_times.
  auto iter_times = node->inputs()[0];

  int input_degree = -1;
  std::vector<TypePtr> input_refine_types;
  for (int i = 1; i < node->inputs().size(); i++) {
    auto input_ = node->input(i);
    functor_op->addInput(input_);
    // get list size
    if (refine_types.count(input_)) {
      auto type = refine_types[input_];
      AT_ASSERT(type->kind() == c10::TypeKind::TupleType,
                "ParallelMap input(,", input_->debugName(), ") must be union type by now, but get "
                , type->annotation_str());
      auto union_type = getUnifiedElementType(type);
      input_refine_types.push_back(union_type);
      if (input_degree != -1) {
        AT_ASSERT(input_degree == type->cast<c10::TupleType>()->elements().size(),
                  "Input List in ParallelMap must have same input_degree!");
      } else {
        input_degree = type->cast<c10::TupleType>()->elements().size();
      }
    }
  }

  functor_op->tys_(c10::tssa::input_refine_types, input_refine_types);
  functor_op->i_(c10::tssa::parallel_degree, input_degree);

  for (auto output : node->outputs()) {
    auto functor_op_output = functor_op->addOutput();
    functor_op_output->copyMetadata(output);
  }

  std::unordered_map<Value*, Value*> values_map;

  std::unordered_set<Value*> parallel_map_block_args;

  for (auto input_ : node->blocks()[0]->inputs()) {
    parallel_map_block_args.insert(input_);
  }

  // Get Fusion Group in ParallelMap
  auto fusion_group = node->blocks()[0]->nodes().front();

  std::unordered_set<Value*> parallelled_args;
  for (auto input_ : node->blocks()[0]->inputs())
    parallelled_args.insert(input_);

  std::vector<int64_t> is_parallelled_args;

  for (int i = fusion_group->inputs().size() - 1; i >= 0; i--) {
    auto input_ = fusion_group->input(i);
    if (!parallelled_args.count(input_)) {
      functor_op->insertInput(i, input_);
      is_parallelled_args.push_back(0);
    } else
      is_parallelled_args.push_back(1);
  }

  std::reverse(is_parallelled_args.begin(), is_parallelled_args.end());
  functor_op->is_(c10::tssa::is_parallelled_args, is_parallelled_args);

  for (auto input_ : fusion_group->blocks()[0]->inputs()) {
    auto subgraph_input = subgraph->addInput();
    subgraph_input->copyMetadata(input_);
    values_map[input_] = subgraph_input;
  }

  for (auto fusion_group_node : fusion_group->blocks()[0]->nodes()) {
    Node* in_graph = subgraph->createClone(fusion_group_node, [&](Value* k) -> Value* {
      if (k->node()->owningBlock() == fusion_group->blocks()[0]) return values_map[k];
      if (k->node()->kind() == prim::Constant) {
        Node* constant = subgraph->createClone(k->node(), [](Value* k){ return nullptr;});
        constant->insertBefore(subgraph->nodes().front());
        return constant->output();
      }
      return k; });
    subgraph->insertNode(in_graph);
    for (int i = 0; i < in_graph->outputs().size(); i++) {
      values_map[fusion_group_node->output(i)] = in_graph->output(i);
    }
  }

  for (auto output : fusion_group->blocks()[0]->outputs()) {
    subgraph->registerOutput(values_map[output]);
  }
  
  SolveUpdate(subgraph);
  return functor_op;
}

void MapFunctorToParallizationBlock(Block* block,
                                    std::unordered_map<Value *, TypePtr>& refine_types) {
  for (auto node = block->nodes().front(); node != block->nodes().back();) {
    if (node->kind() == c10::prim::ParallelMap) {
      auto parallelled_functor_op = GetParallelledFunctorByParallelMap(node, refine_types);
      replace(node, parallelled_functor_op);
      node = parallelled_functor_op;
    } else {
      for (auto node_block : node->blocks()) {
        MapFunctorToParallizationBlock(node_block, refine_types);
      }
    }
    node = node->next();
  }
}

void MapFunctorToParallization(const std::shared_ptr<Graph>& graph,
                               std::unordered_map<Value *, TypePtr>& refine_types) {
  MapFunctorToParallizationBlock(graph->block(), refine_types);
}

Operation CreateTeOperator(const Node* node) {
  auto graph_builder = std::make_shared<GraphBuilder>(node);
  return [graph_builder](Stack& stack) -> int {
    graph_builder->run(stack);
    return 0;
  };
}

RegisterOperators ParallelOps({
  torch::jit::Operator(
    c10::tssa::ParallelledFunctor,
    CreateTeOperator,
    AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
});

}  // namespace jit
}  // namespace torch



