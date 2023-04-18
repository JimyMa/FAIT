#include "fuse_ops.h"

#include <torch/csrc/jit/tensorexpr/operators/operators.h>

#include "fuser/nnc_func.h"
#include "fuser/tssa_set_ops.h"
#include "parallelize_loops.h"
#include "tensor_ssa.h"
#include "type_utils.h"
#include "util/disjoint_set.h"
#include "util/ir.h"
#include "util/traits.h"

namespace torch {
namespace jit {

static auto registry = registerTssaSetOps();

OperatorSet fusableOps{
    "aten::tensor.bool(bool t, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
    "aten::tensor.float(float t, *, ScalarType? dtype=None, Device? "
    "device=None, bool requires_grad=False) -> Tensor",
    "aten::tensor.int(int t, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
    "aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=0) -> "
    "Tensor(a)",
    "aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool "
    "non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> "
    "Tensor(a)",
    "aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? "
    "layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
    "aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, "
    "Layout? layout=None, Device? device=None, bool? pin_memory=None) -> "
    "Tensor",
    "aten::exp(Tensor self) -> Tensor",
    "aten::sigmoid(Tensor self) -> Tensor",
    "aten::tril(Tensor self, int diagonal=0) -> Tensor",
    "aten::triu(Tensor self, int diagonal=0) -> Tensor",
    "aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor",
    "aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> "
    "Tensor",
    "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
    "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
    "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::div.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::div.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::minimum(Tensor self, Tensor other) -> Tensor",
    "aten::maximum(Tensor self, Tensor other) -> Tensor",
    "aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::eq.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::eq.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::ne.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::ne.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::lt.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::le.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::le.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::gt.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::gt.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::ge.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::ge.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor "
    "values, Tensor indices)",
    "aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)",
    "aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? "
    "end=None, SymInt step=1) -> Tensor(a)",
    "aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)",
    "aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)",
    "aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)",
    "aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> "
    "Tensor(a)",
    "aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)",
    "aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)",
    "aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)",
    "aten::cat(Tensor[] tensors, int dim=0) -> Tensor",
    "aten::stack(Tensor[] tensors, int dim=0) -> Tensor",
    "aten::repeat(Tensor self, SymInt[] repeats) -> Tensor",
    // "aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor",
    "aten::size.int(Tensor self, int dim) -> int",
    "aten::__getitem__.t(t[](a) list, int idx) -> t(*)",
    "prim::TupleUnpack(Any tup) -> ...",
    "tssa::Assign(Tensor self, Tensor src) -> Tensor",
};

static std::unordered_set<Symbol> fusableNoOpSymbols{
    tssa::Update, prim::ListConstruct, prim::ListUnpack};

static std::unordered_set<Symbol> workingSymbols{
    // Tensor creation/conversion
    aten::tensor, aten::arange, aten::zeros, aten::to, aten::contiguous,
    // Elementwise
    aten::exp, aten::sigmoid, aten::clamp, aten::tril, aten::triu,
    // Binary
    aten::add, aten::sub, aten::mul, aten::div, aten::__and__, aten::minimum,
    aten::maximum,
    // Comparison
    aten::eq, aten::ne, aten::lt, aten::le, aten::gt, aten::ge,
    // Reduction
    aten::max, aten::softmax,
    // Copy
    aten::repeat, aten::cat, aten::stack, aten::index,
    // TensorSSA
    tssa::Assign};

static std::unordered_map<Symbol, bool (*)(Node *node)> fusabilityCheckers{
    {aten::__getitem__,
     [](Node *node) {
       return node->owningBlock() == node->input(0)->node()->owningBlock();
     }},
    {aten::max,
     [](Node *node) {
       return node->outputs().size() == 1 || !node->output(1)->hasUses();
     }},
    {aten::cat,
     [](Node *node) {
       return node->input(0)->node()->kind() != prim::ParallelMap;
     }},
    {aten::stack,
     [](Node *node) {
       return node->input(0)->node()->kind() != prim::ParallelMap;
     }},
    {aten::index,
     [](Node *node) {
       auto indices = node->input(1)->node()->inputs();
       if (indices.front()->type()->cast<TensorType>()->scalarType() !=
           c10::kLong)  // static shape only when indices are `long`
         return false;
       auto output = node->output(0);
       if (std::any_of(
               output->uses().begin(), output->uses().end(),
               [](const Use &use) { return use.user->kind() == tssa::Assign; }))
         return false;  // do not fuse workaround for `aten::index_put_`
       return true;
     }},
    {tssa::Assign,
     [](Node *node) { return node->input(0)->node()->kind() != aten::index; }},
};

static bool isFusable(Node *node, bool isOut,
                      const std::unordered_set<Node *> fusedNodes) {
  // Check if the symbol is fusable
  auto kind = node->kind();
  auto op = node->maybeOperator();
  if (op) {
    if (!node->isMemberOf(fusableOps)) return false;
  } else {
    if (!fusableNoOpSymbols.count(kind)) return false;
  }

  // Do not fuse nodes with mutated inputs or outputs
  for (auto input : node->inputs()) {
    if (isMutated(input)) return false;
  }
  for (auto output : node->outputs()) {
    if (isMutated(output)) return false;
  }

  if (isOut) {
    // Fused subgraphs cannot have non-tensor outputs
    for (auto output : node->outputs()) {
      if (!output->type()->castRaw<TensorType>()) return false;
    }
  } else {
    for (auto output : node->outputs()) {
      if (output->type()->kind() != TypeKind::TensorType &&
          std::any_of(
              output->uses().begin(), output->uses().end(),
              [&](const Use &use) { return !fusedNodes.count(use.user); }))
        return false;
    }
  }

  // Perform addtional checking
  if (fusabilityCheckers.count(kind))
    return fusabilityCheckers[kind](node);
  else
    return true;
}

static bool shouldFuseGroup(Node *head, Node *tail,
                            const std::unordered_set<Node *> fusedNodes) {
  // Check number of working nodes
  size_t numWorking = 0;
  for (auto node = head; node != tail; node = node->next())
    numWorking += workingSymbols.count(node->kind());
  if (numWorking <= 1) return false;

  // Check if any constant is updated
  for (auto node = head; node != tail; node = node->next()) {
    if (node->kind() == tssa::Update &&
        node->input(0)->node()->kind() == prim::Constant)
      return false;
  }

  return true;
}

static void findGroupInOutValues(Node *head, Node *tail,
                                 std::vector<Value *> &inputs,
                                 std::vector<Value *> &outputs) {
  // Create group node set
  std::unordered_set<Node *> groupNodeSet;
  for (auto node = head; node != tail; node = node->next())
    groupNodeSet.insert(node);

  // Find inputs and outputs for each node
  for (auto node = head; node != tail; node = node->next()) {
    // Find inputs
    for (auto input : node->inputs()) {
      // Constants are not considered inputs
      auto defNode = input->node();
      if (defNode->kind() == prim::Constant) continue;

      // Skip if this input is defined by a node in the group
      if (groupNodeSet.count(defNode)) continue;

      // Skip if added before
      if (std::find(inputs.begin(), inputs.end(), input) != inputs.end())
        continue;

      // Add to input list
      inputs.push_back(input);
    }

    // Find outputs
    for (auto output : node->outputs()) {
      // Skip if all of its uses are in the group
      auto &uses = output->uses();
      if (std::all_of(uses.begin(), uses.end(), [&](const Use &use) {
            return groupNodeSet.count(use.user);
          }))
        continue;

      // Add to output list
      outputs.push_back(output);
    }
  }
}

Node *commitFusion(Node *head, Node *tail, Graph *graph,
                   ValueTypeMap &refinedTypes) {
  // Collect input and output values from the nodes in the group
  std::vector<Value *> inputs, outputs;
  findGroupInOutValues(head, tail, inputs, outputs);

  // Create fusion node
  auto fusionNode = graph->create(prim::FusionGroup, inputs, 0);
  fusionNode->insertBefore(tail);
  auto fusionBlock = fusionNode->addBlock();

  // Replace outputs of the group
  for (auto output : outputs) {
    auto groupOut = fusionNode->addOutput()->setType(output->type());
    transferRefinedType(output, groupOut, refinedTypes);
    output->replaceAllUsesAfterNodeWith(fusionNode, groupOut);
  }

  // Map input values
  std::unordered_map<Value *, Value *> valueMap;
  for (auto input : inputs) {
    auto param = fusionBlock->addInput()->setType(input->type());
    transferRefinedType(input, param, refinedTypes);
    valueMap.insert({input, param});
  }

  // Move nodes to the new block
  moveNodesToBlock(head, fusionNode, fusionBlock, valueMap, &refinedTypes);
  removeDeadRefinedTypes(refinedTypes, graph);

  // Handle block returns
  for (auto output : outputs) {
    fusionBlock->insertOutput(fusionBlock->outputs().size(),
                              valueMap.at(output));
  }

  return fusionNode;
}

static void fuseOpsIn(Block *block, Graph *graph, ValueTypeMap &refinedTypes) {
  // Find tail node to begin with
  for (auto tail = block->return_node(), head = tail;
       tail != block->nodes().front(); tail = head) {
    // Record known predecessors
    std::unordered_set<Node *> knownPreds, fusedNodes;

    // Traverse in reverse order to find all fusable nodes
    for (auto node = head->prev(); node != block->param_node();
         node = node->prev()) {
      // Do not fuse constants
      auto fixRangeIfEmpty = [&]() {
        if (head == tail) tail = head = node;
      };
      if (node->kind() == prim::Constant) {
        fixRangeIfEmpty();
        continue;
      }

      // Check if current node is fusable
      if (!isFusable(node, !knownPreds.count(node), fusedNodes)) {
        fixRangeIfEmpty();
        continue;
      }

      // Check if this node can be moved to the group
      bool canMove = true;
      for (auto output : node->outputs()) {
        for (auto &use : output->uses()) {
          if (use.user->isBefore(head)) {
            canMove = false;
            break;
          }
        }
        if (!canMove) break;
      }
      if (!canMove) {
        fixRangeIfEmpty();
        continue;
      }

      // Add predecessors to map
      for (auto input : node->inputs()) knownPreds.insert(input->node());

      // Move this node to the head of the group
      node->moveBefore(head);
      fusedNodes.insert(node);
      head = node;
    }

    // Check if current group can be fused
    if (!shouldFuseGroup(head, tail, fusedNodes)) continue;

    // Commit fusion
    head = commitFusion(head, tail, graph, refinedTypes);
  }
}

void FuseOps(const std::shared_ptr<Graph> &graph, ValueTypeMap &refinedTypes) {
  // Collect all blocks
  std::vector<Block *> blocks;
  blocks.push_back(graph->block());
  traversePreOrder(graph->block(), [&](Node *node) {
    for (auto block : node->blocks()) blocks.push_back(block);
    return true;
  });

  // Fuse operators inside blocks
  for (auto block : blocks) fuseOpsIn(block, graph.get(), refinedTypes);
}

void printOpsInFusionGroups(const std::shared_ptr<Graph> &graph) {
  std::unordered_set<std::string> schemas;
  traversePreOrder(graph->block(), [&](Node *node) {
    auto parent = node->owningBlock()->owningNode();
    if (!parent || parent->kind() != prim::FusionGroup) return true;
    if (!node->maybeSchema()) return true;
    std::stringstream ss;
    ss << node->schema();
    auto schema = ss.str();
    if (schemas.count(schema)) return true;
    auto hasShapeFunc = node->isMemberOf(tensorexpr::identicalShapeOps) ||
                        node->isMemberOf(tensorexpr::shapeFuncs);
    auto hasComputeFunc = node->isMemberOf(tensorexpr::customLoweringFuncs) ||
                          bool(tensorexpr::getStandardLoweringFor(schema));
    print(std::cout, schema, ' ', hasShapeFunc, ' ', hasComputeFunc, '\n');
    schemas.insert(ss.str());
    return true;
  });
}

}  // namespace jit
}  // namespace torch
