//
// Created by jimyma on 1/27/23.
//
#include "passes/te_op.h"

#include <torch/csrc/jit/passes/constant_pooling.h>

#include <memory>

#include "fuser/graph_builder.h"
#include "fuser/solve_update.h"
#include "passes/common_passes.h"
#include "passes/parallelize_loops.h"
#include "passes/type_utils.h"
#include "torch/csrc/jit/runtime/custom_operator.h"
#include "util/ir.h"

namespace torch {
namespace jit {

struct ParallelFunctor {
  ParallelFunctor(Node* functor_op, Value* idx)
      : functor_op_(functor_op), idx_(idx) {}
  Node* functor_op_;
  Value* idx_;
};

void runPasses(const std::shared_ptr<Graph>& graph) {
  SolveUpdate(graph);
  EliminateDeadCodeTSSA(graph);
  ConstantPooling(graph);
}

static Node* GetParallelledFunctorByParallelMap(
    Node* node, std::unordered_map<Value*, TypePtr>& refine_types) {
  // Construct an op
  auto functor_op =
      node->owningGraph()->createWithSubgraph(c10::tssa::ParallelledFunctor);

  auto subgraph = functor_op->g(attr::Subgraph);

  auto input_degree = *constant_as<int64_t>(node->input(0));
  std::vector<TypePtr> input_refine_types;

  for (int i = 1; i < node->inputs().size(); i++) {
    auto input_ = node->input(i);
    functor_op->addInput(input_);
    // get list size
    if (refine_types.count(input_)) {
      auto type = refine_types[input_];
      AT_ASSERT(type->kind() == c10::TypeKind::TupleType, "ParallelMap input(,",
                input_->debugName(), ") must be union type by now, but get ",
                type->annotation_str());
      auto union_type = getUnifiedElementType(type);
      input_refine_types.push_back(union_type);
      AT_ASSERT(input_degree == type->cast<c10::TupleType>()->elements().size(),
                "Input List in ParallelMap must have same input_degree!");
    }
  }

  functor_op->tys_(c10::tssa::input_refine_types, input_refine_types);
  functor_op->i_(c10::tssa::parallel_degree, input_degree);
  functor_op->i_(c10::tssa::is_parallel_map, true);

  for (auto output : node->outputs()) {
    auto functor_op_output = functor_op->addOutput();
    functor_op_output->copyMetadata(output);
    transferRefinedType(output, functor_op_output, refine_types);
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

  for (int i = fusion_group->inputs().size() - 1; i > 0; i--) {
    auto input_ = fusion_group->input(i);
    if (!parallelled_args.count(input_)) {
      functor_op->insertInput(i, input_);
      is_parallelled_args.push_back(0);
    } else
      is_parallelled_args.push_back(1);
  }

  std::reverse(is_parallelled_args.begin(), is_parallelled_args.end());
  functor_op->is_(c10::tssa::is_parallelled_args, is_parallelled_args);

  for (int i = 0; i < fusion_group->blocks()[0]->inputs().size(); i++) {
    auto input_ = fusion_group->blocks()[0]->inputs()[i];
    auto subgraph_input = subgraph->addInput();
    subgraph_input->copyMetadata(input_);
    values_map[input_] = subgraph_input;
  }

  for (auto fusion_group_node : fusion_group->blocks()[0]->nodes()) {
    Node* in_graph =
        subgraph->createClone(fusion_group_node, [&](Value* k) -> Value* {
          if (k->node()->owningBlock() == fusion_group->blocks()[0])
            return values_map[k];
          if (k->node()->kind() == prim::Constant) {
            Node* constant = subgraph->createClone(
                k->node(), [](Value* k) { return nullptr; });
            constant->insertBefore(subgraph->nodes().front());
            return constant->output();
          }
          return k;
        });
    subgraph->insertNode(in_graph);
    for (int i = 0; i < in_graph->outputs().size(); i++) {
      values_map[fusion_group_node->output(i)] = in_graph->output(i);
    }
  }

  for (auto output : fusion_group->blocks()[0]->outputs()) {
    subgraph->registerOutput(values_map[output]);
  }

  runPasses(subgraph);

  return functor_op;
}

static void MapFunctorToParallizationBlock(
    Block* block, std::unordered_map<Value*, TypePtr>& refine_types) {
  for (auto node = block->nodes().front(); node != block->nodes().back();) {
    if (node->kind() == c10::prim::ParallelMap) {
      auto parallelled_functor_op =
          GetParallelledFunctorByParallelMap(node, refine_types);
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

void MapFunctorToParallization(
    const std::shared_ptr<Graph>& graph,
    std::unordered_map<Value*, TypePtr>& refine_types) {
  MapFunctorToParallizationBlock(graph->block(), refine_types);
}

static Node* GetParallelledFunctorByFusedOp(
    Node* node, std::unordered_map<Value*, TypePtr>& refine_types) {
  // Construct an op
  auto functor_op =
      node->owningGraph()->createWithSubgraph(c10::tssa::ParallelledFunctor);

  auto subgraph = functor_op->g(attr::Subgraph);

  int input_degree = 1;
  std::vector<TypePtr> input_refine_types;
  for (int i = 0; i < node->inputs().size(); i++) {
    auto input_ = node->input(i);
    functor_op->addInput(input_);
    // get list size
    if (refine_types.count(input_)) {
      auto type = refine_types[input_];
      input_refine_types.push_back(type);
    } else {
      input_refine_types.push_back(input_->type());
    }
  }

  functor_op->tys_(c10::tssa::input_refine_types, input_refine_types);
  functor_op->i_(c10::tssa::parallel_degree, input_degree);
  functor_op->i_(c10::tssa::is_parallel_map, false);

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
  //   auto fusion_group = node->blocks()[0]->nodes().front();

  std::unordered_set<Value*> parallelled_args;
  for (auto input_ : node->blocks()[0]->inputs())
    parallelled_args.insert(input_);

  std::vector<int64_t> is_parallelled_args(node->blocks()[0]->inputs().size(),
                                           1);
  functor_op->is_(c10::tssa::is_parallelled_args, is_parallelled_args);

  // placeholder for axis
  subgraph->addInput();
  for (auto input_ : node->blocks()[0]->inputs()) {
    auto subgraph_input = subgraph->addInput();
    subgraph_input->copyMetadata(input_);
    values_map[input_] = subgraph_input;
  }
  for (auto fusion_group_node : node->blocks()[0]->nodes()) {
    Node* in_graph =
        subgraph->createClone(fusion_group_node, [&](Value* k) -> Value* {
          if (k->node()->owningBlock() == node->blocks()[0])
            return values_map[k];
          if (k->node()->kind() == prim::Constant) {
            Node* constant = subgraph->createClone(
                k->node(), [](Value* k) { return nullptr; });
            constant->insertBefore(subgraph->nodes().front());
            return constant->output();
          }
          return k;
        });
    subgraph->insertNode(in_graph);
    for (int i = 0; i < in_graph->outputs().size(); i++) {
      values_map[fusion_group_node->output(i)] = in_graph->output(i);
    }
  }
  for (auto output : node->blocks()[0]->outputs()) {
    subgraph->registerOutput(values_map[output]);
  }

  runPasses(subgraph);

  return functor_op;
}

static void FusedOpToParallizationBlock(
    Block* block, std::unordered_map<Value*, TypePtr>& refine_types) {
  for (auto node = block->nodes().front(); node != block->nodes().back();) {
    if (node->kind() == prim::FusionGroup) {
      auto parallelled_functor_op =
          GetParallelledFunctorByFusedOp(node, refine_types);
      // print(std::cout, parallelled_functor_op->schema());
      replace(node, parallelled_functor_op);
      node = parallelled_functor_op;
    } else {
      for (auto node_block : node->blocks()) {
        FusedOpToParallizationBlock(node_block, refine_types);
      }
    }
    node = node->next();
  }
}

void FusedOpToParallization(const std::shared_ptr<Graph>& graph,
                            std::unordered_map<Value*, TypePtr>& refine_types) {
  // std::cout << "before fuse: " << std::endl;
  // graph->dump();
  FusedOpToParallizationBlock(graph->block(), refine_types);
  //   std::cout << "after fuse: " << std::endl;
  //   graph->dump();
}

static Operation CreateTeOperator(const Node* node) {
  auto graph_builder = std::make_shared<GraphBuilder>(node);
  return [graph_builder](Stack& stack) -> int {
    graph_builder->run(stack);
    return 0;
  };
}

RegisterOperators ParallelOps({
    torch::jit::Operator(c10::tssa::ParallelledFunctor, CreateTeOperator,
                         AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
});

}  // namespace jit
}  // namespace torch
