#include "tensor_ssa.h"

#include <torch/csrc/jit/ir/alias_analysis.h>

#include "util/disjoint_set.h"
#include "util/ir.h"
#include "util/op_traits.h"

namespace c10 {
namespace tssa {

auto ns = Symbol::fromQualString("namespaces::tssa");
auto Assign = Symbol::fromQualString("tssa::Assign");
auto Update = Symbol::fromQualString("tssa::Update");

}  // namespace tssa
}  // namespace c10

namespace torch {
namespace jit {

inline static Node *rewriteMutating(
    Node *node, Graph *graph, DisjointSet<Value *> &aliasSets,
    std::unordered_map<Value *, std::vector<Node *>> &mutRecords) {
    // Replace mutating operations with non-mutating ones
    auto block = node->owningBlock();
    auto beforeAssign = node->input(0);
    Node *assignNode = nullptr;
    switch (node->kind()) {
        case aten::copy_: {
            assignNode = createTssaAssign(graph, beforeAssign, node->input(1));
            replace(node, assignNode);
            break;
        }

        case aten::index_put_: {
            // Create imaginary advanced indexing view
            auto indexNode =
                graph->create(aten::index, {node->input(0), node->input(1)});
            indexNode->insertAfter(node);
            aliasSets.merge(indexNode->input(0), indexNode->output(0));

            // Create assignment to the imaginary view
            beforeAssign = indexNode->output(0);
            assignNode = createTssaAssign(graph, beforeAssign, node->input(2));
            assignNode->insertAfter(indexNode);
            TORCH_CHECK(!node->hasUses());
            node->destroy();
            break;
        }

        default: {
            // Create immutable operation node
            auto mutSym = node->kind();
            std::string immutOpName(mutSym.toUnqualString());
            immutOpName.pop_back();
            auto immutSym = Symbol::fromQualString(
                std::string(mutSym.ns().toUnqualString()) + "::" + immutOpName);
            auto opNode = graph->create(immutSym, node->inputs());
            opNode->insertBefore(node);
            TORCH_INTERNAL_ASSERT(opNode->maybeSchema());

            // Create assignment node
            assignNode =
                createTssaAssign(graph, beforeAssign, opNode->output(0));
            replace(node, assignNode);
            break;
        }
    }
    TORCH_INTERNAL_ASSERT(assignNode);

    // Update aliases of the assigned value
    auto afterAssign = assignNode->output(0);
    auto lastNode = assignNode;
    for (auto alias : aliasSets.getSetOf(beforeAssign)) {
        auto mutNode = assignNode;
        if (alias != beforeAssign) {
            auto updateNode = createTssaUpdate(graph, alias, afterAssign);
            updateNode->insertAfter(lastNode);
            mutNode = updateNode;
            lastNode = updateNode;
        }
        if (alias->node()->owningBlock() != block) {
            // Alias is not defined in current block, add to mutation record
            if (mutRecords.count(alias))
                mutRecords[alias].push_back(mutNode);
            else
                mutRecords[alias] = {mutNode};
        }
    }

    return lastNode;
}

void ToTensorSSA(const std::shared_ptr<Graph> &graph) {
    // Find all mutated tensors and remove mutation
    DisjointSet<Value *> aliasSets;
    std::unordered_map<Value *, std::vector<Node *>> mutRecords;
    rewriteNode(graph->block(), [&](Node *node) -> Node * {
        // Skip non-tensor operations
        if (node->inputs().empty() || node->outputs().empty()) return nullptr;
        if (node->input(0)->type()->kind() != TypeKind::TensorType ||
            node->output(0)->type()->kind() != TypeKind::TensorType)
            return nullptr;

        // Rewrite mutating nodes to remove mutation
        if (isMutating(node))
            return rewriteMutating(node, graph.get(), aliasSets, mutRecords);

        // Extend tensor alias graph if the node is aliasing
        if (isAliasing(node)) aliasSets.merge(node->input(0), node->output(0));

        return nullptr;
    });

    return;
}

}  // namespace jit
}  // namespace torch
