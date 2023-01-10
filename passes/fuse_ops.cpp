#include "fuse_ops.h"

#include "passes/tensor_ssa.h"
#include "util/disjoint_set.h"
#include "util/ir.h"

namespace torch {
namespace jit {

OperatorSet fusableOps{
    "aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, "
    "Device? device=None, bool? pin_memory=None) -> Tensor",
    "aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, "
    "Layout? layout=None, Device? device=None, bool? pin_memory=None) -> "
    "Tensor",
    "aten::arange.start_step(Scalar start, Scalar end, Scalar step=1, *, "
    "ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? "
    "pin_memory=None) -> Tensor",
    "aten::exp(Tensor self) -> Tensor",
    "aten::log(Tensor self) -> Tensor",
    "aten::sin(Tensor self) -> Tensor",
    "aten::cos(Tensor self) -> Tensor",
    "aten::sqrt(Tensor self) -> Tensor",
    "aten::sigmoid(Tensor self) -> Tensor",
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
    "aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)",
    "aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? "
    "end=None, SymInt step=1) -> Tensor(a)",
    "aten::squeeze(Tensor(a) self) -> Tensor(a)",
    "aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)",
    "aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)",
    "aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)",
    "aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)",
    "aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> "
    "Tensor(a)",
    "aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)",
    "aten::concat(Tensor[] tensors, int dim=0) -> Tensor",
    "aten::stack(Tensor[] tensors, int dim=0) -> Tensor",
    "aten::repeat(Tensor self, SymInt[] repeats) -> Tensor",
    "aten::size.int(Tensor self, int dim) -> int",
    "aten::size(Tensor self) -> int[]",
    "aten::__getitem__.t(t[](a) list, int idx) -> t(*)"};

static std::vector<Symbol> fusableAtenSymbols{
    // Tensor creation
    aten::arange,
    // Elementwise
    aten::exp, aten::log, aten::sin, aten::cos, aten::sqrt, aten::sigmoid,
    // Binary
    aten::add, aten::sub, aten::mul, aten::div, aten::minimum, aten::maximum,
    // View
    aten::select, aten::slice, aten::squeeze, aten::unsqueeze, aten::reshape,
    aten::view, aten::expand, aten::permute,
    // Copy
    aten::repeat, aten::concat, aten::stack,
    // Auxiliary
    aten::size, aten::__getitem__};

static std::unordered_set<Symbol> fusablePrimSymbols{prim::ListConstruct};

static std::unordered_map<Symbol, bool (*)(Node *node)> fusabilityCheckers{
    {aten::__getitem__,
     [](Node *node) {
         auto listTy = node->input(0)->type()->castRaw<ListType>();
         return !listTy->getElementType()->castRaw<TensorType>();
     }},
};

static void printFusableOps() {
    for (auto sym : fusableAtenSymbols) {
        for (auto &op : getAllOperatorsFor(sym))
            std::cout << '"' << op->schema() << "\",\n";
    }
}

static bool isFusable(Node *node) {
    // Check if the symbol is fusable
    auto kind = node->kind();
    auto op = node->maybeOperator();
    if (op) {
        if (!node->isMemberOf(fusableOps)) return false;
    } else {
        if (!fusablePrimSymbols.count(kind)) return false;
    }

    // Perform addtional checking
    if (fusabilityCheckers.count(kind))
        return fusabilityCheckers[kind](node);
    else
        return true;
}

static std::unordered_set<Symbol> workingSymbols{
    // Tensor creation
    aten::arange,
    // Elementwise
    aten::exp, aten::log, aten::sin, aten::cos, aten::sqrt, aten::sigmoid,
    // Binary
    aten::add, aten::sub, aten::mul, aten::div, aten::minimum, aten::maximum,
    // Copy
    aten::repeat, aten::concat, aten::stack};

void addTssaSymbols() {
    fusablePrimSymbols.insert({tssa::Assign, tssa::Update});
    workingSymbols.insert(tssa::Assign);
    fusabilityCheckers.insert({tssa::Assign, [](Node *node) {
                                   return node->input(0)->node()->kind() !=
                                          aten::index;
                               }});
}

static bool shouldFuse(const std::vector<Node *> group) {
    // No need to fuse single-node group
    if (group.size() == 1) return false;

    // Reject group that has no working symbols
    if (std::none_of(group.begin(), group.end(), [](Node *node) {
            return workingSymbols.count(node->kind());
        }))
        return false;

    return true;
}

static void findGroupInOutValues(const std::vector<Node *> &group,
                                 std::vector<Value *> &inputs,
                                 std::vector<Value *> &outputs) {
    std::unordered_set<Node *> groupNodeSet(group.begin(), group.end());
    for (auto node : group) {
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

static bool compareNodePosition(Node *lhs, Node *rhs) {
    return lhs->isBefore(rhs);
}

static void moveBeforeRecursively(Node *node, Node *pivot) {
    if (node->owningBlock() != pivot->owningBlock()) return;
    if (node->isBefore(pivot)) return;
    for (auto input : node->inputs()) {
        moveBeforeRecursively(input->node(), pivot);
        node->moveBefore(pivot);
    }
}

static void commitFusion(std::vector<Node *> &&group, Graph *graph) {
    // Collect input and output values from the nodes in the group
    std::vector<Value *> inputs, outputs;
    findGroupInOutValues(group, inputs, outputs);

    // Move all nodes whose values are used by the fusion group before the front
    // of this group
    auto frontNode = group.front();
    for (auto node : group) {
        for (auto input : node->inputs()) {
            // constants need special care, as they are not considered inputs of
            // the group
            auto defNode = input->node();
            if (defNode->kind() == prim::Constant &&
                defNode->isAfter(frontNode))
                defNode->moveBefore(frontNode);
        }
    }
    for (auto input : inputs) moveBeforeRecursively(input->node(), frontNode);

    // Create fusion node
    auto fusionNode = graph->create(prim::FusionGroup, inputs, outputs.size());
    fusionNode->insertBefore(frontNode);
    auto block = fusionNode->addBlock();

    // Map input values
    std::unordered_map<Value *, Value *> valueMap;
    for (auto input : inputs) {
        auto param = block->addInput();
        param->setType(input->type());
        valueMap.insert({input, param});
    }

    // Move nodes to the new block
    for (auto node : group) {
        node->moveBefore(block->return_node());
        for (auto i = 0u; i < node->inputs().size(); i++) {
            auto arg = node->input(i);
            if (valueMap.count(arg)) node->replaceInput(i, valueMap[arg]);
        }
    }

    // Handle block return and node outputs
    for (auto i = 0u; i < outputs.size(); i++) {
        auto output = outputs[i];
        block->return_node()->addInput(output);
        fusionNode->output(i)->setType(output->type());
        output->replaceAllUsesAfterNodeWith(fusionNode, fusionNode->output(i));
    }
}

static void fuseOpsIn(Block *block, Graph *graph) {
    // Find fusable nodes and create disjoint sets for nodes
    std::unordered_set<Node *> fusableNodeSet;
    DisjointSet<Node *> fusionDisjSets;
    auto checkFusable = [&](Node *node) {
        if (fusableNodeSet.count(node)) return true;
        if (isFusable(node)) {
            fusableNodeSet.insert(node);
            return true;
        } else
            return false;
    };
    for (auto node : block->nodes().reverse()) {
        // Check fusability of this node
        if (!checkFusable(node)) continue;

        // Check fusability of predecessors and possibly merge nodes
        for (auto input : node->inputs()) {
            auto pred = input->node();
            if (pred->owningBlock() != block)
                continue;  // cannot fuse nodes outside this block
            if (checkFusable(pred)) fusionDisjSets.merge(node, pred);
        }
    }

    // Check group fusability and commit fusion
    std::vector<Node *> fusableNodeList(fusableNodeSet.begin(),
                                        fusableNodeSet.end());
    fusableNodeSet.clear();
    std::sort(fusableNodeList.begin(), fusableNodeList.end(),
              compareNodePosition);
    std::unordered_set<Node *> checkedNodes;
    for (auto node : fusableNodeList) {
        // Skip if checked before
        if (checkedNodes.count(node)) continue;

        // Check if the group should be fused
        auto group = fusionDisjSets.getSetOf(node);
        if (!shouldFuse(group)) continue;
        std::sort(group.begin(), group.end(), compareNodePosition);
        for (auto nodeInGroup : group) checkedNodes.insert(nodeInGroup);

        // Commit fusion
        commitFusion(std::move(group), graph);
    }
}

void FuseOps(const std::shared_ptr<Graph> &graph) {
    // Add TensorSSA symbols to records
    addTssaSymbols();

    // Collect all blocks
    std::vector<Block *> blocks;
    blocks.push_back(graph->block());
    traverse(graph->block(), [&](Node *node) {
        for (auto block : node->blocks()) blocks.push_back(block);
        return true;
    });

    // Fuse operators inside blocks
    for (auto block : blocks) fuseOpsIn(block, graph.get());
    // printFusableOps();
}

}  // namespace jit
}  // namespace torch
