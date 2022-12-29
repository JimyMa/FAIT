#include "tensor_ssa.h"

#include <torch/csrc/jit/ir/alias_analysis.h>

#include "util/disjoint_set.h"
#include "util/ir.h"
#include "util/traits.h"

namespace c10 {
namespace tssa {

auto ns = Symbol::fromQualString("namespaces::tssa");
auto Assign = Symbol::fromQualString("tssa::Assign");
auto Update = Symbol::fromQualString("tssa::Update");

}  // namespace tssa
}  // namespace c10

namespace torch {
namespace jit {

static Node *rewriteMutating(
    Node *node, Graph *graph, DisjointSet<Value *> &aliasSets,
    std::vector<Value *> &mutValues,
    std::unordered_map<Value *, std::vector<Node *>> &mutNodes) {
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
            if (mutNodes.count(alias))
                mutNodes[alias].push_back(mutNode);
            else {
                mutValues.push_back(alias);
                mutNodes[alias] = {mutNode};
            }
        }
    }

    return lastNode;
}

static void addMutatedValueToBlock(
    Value *mutated, Block *block, std::unordered_set<Block *> &visitedBlocks,
    std::unordered_map<Value *, Value *> &valueToMut, bool handleNode = true) {
    // Skip if this block if visited before
    if (visitedBlocks.count(block)) return;
    visitedBlocks.insert(block);

    // Add to block and node returns
    block->insertOutput(block->outputs().size(), mutated);
    auto node = block->owningNode();
    if (handleNode) {
        auto nodeRet = node->addOutput();
        valueToMut.insert({nodeRet, mutated});
    }

    // Handle values that are specific to node kinds
    switch (node->kind()) {
        case prim::Loop: {
            // add to block parameter of loop body
            auto param = block->addInput();
            valueToMut.insert({param, mutated});
            // add to argument of loop node
            node->addInput(mutated);
            break;
        }

        case prim::If: {
            // add to the block of the other branch
            auto blockId = block == node->blocks()[1];
            addMutatedValueToBlock(mutated, node->blocks()[!blockId],
                                   visitedBlocks, valueToMut, false);
            break;
        }
    }
}

static void renameValues(
    Block *block, std::unordered_map<Value *, Value *> &valueToMut,
    std::unordered_map<Value *, std::vector<Value *>> &renameStacks) {
    // Initialize rename counts in current scope
    std::unordered_map<Value *, size_t> renameCounts;
    auto updateValue = [&](Value *value) {
        // find mutated version of this value
        Value *mutated = nullptr;
        if (valueToMut.count(value))
            mutated = valueToMut[value];
        else {
            auto defNode = value->node();
            auto kind = defNode->kind();
            if (kind == tssa::Assign || kind == tssa::Update) {
                mutated = valueToMut[defNode->input(0)];
                valueToMut.insert({value, mutated});
            }
        }
        if (!mutated) return;
        // add to rename stack
        renameStacks[mutated].push_back(value);
        // add to rename counts
        if (renameCounts.count(mutated))
            renameCounts[mutated]++;
        else
            renameCounts.insert({mutated, 1});
    };
    auto replaceInputsOf = [&](Node *node) {
        for (auto i = 0u; i < node->inputs().size(); i++) {
            auto input = node->input(i);
            if (!renameStacks.count(input)) continue;
            node->replaceInput(i, renameStacks[input].back());
        }
    };

    // Add parameters to rename stack
    for (auto param : block->inputs()) updateValue(param);

    // Process each node
    for (auto node : block->nodes()) {
        // replace inputs
        replaceInputsOf(node);
        // visit owned blocks
        for (auto nested : node->blocks())
            renameValues(nested, valueToMut, renameStacks);
        // update outputs
        for (auto output : node->outputs()) updateValue(output);
    }

    // Process return node
    replaceInputsOf(block->return_node());

    // Restore rename stack
    for (auto &pair : renameCounts) {
        for (auto i = 0u; i < pair.second; i++)
            renameStacks[pair.first].pop_back();
    }
}

void ToTensorSSA(const std::shared_ptr<Graph> &graph) {
    // Find all mutated tensors and remove mutation
    DisjointSet<Value *> aliasSets;
    std::vector<Value *> mutValues;
    std::unordered_map<Value *, std::vector<Node *>> mutNodes;
    rewrite(graph->block(), [&](Node *node) -> Node * {
        // Skip non-tensor operations
        if (node->inputs().empty() || node->outputs().empty()) return nullptr;
        if (node->input(0)->type()->kind() != TypeKind::TensorType ||
            node->output(0)->type()->kind() != TypeKind::TensorType)
            return nullptr;

        // Rewrite mutating nodes to remove mutation
        if (isMutating(node)) {
                        return rewriteMutating(node, graph.get(), aliasSets, mutValues,
                                   mutNodes);
        }

        // Extend tensor alias graph if the node is aliasing
        if (isAliasing(node)) aliasSets.merge(node->input(0), node->output(0));

        return nullptr;
    });

    // Add block parameters and returns for out-of-block mutation
    std::unordered_map<Value *, Value *> valueToMut;
    for (auto mutated : mutValues) {
        valueToMut.insert({mutated, mutated});
        auto defBlock = mutated->node()->owningBlock();
        std::unordered_set<Block *> visitedBlocks;
        auto &nodes = mutNodes[mutated];
        for (auto node : nodes) {
            for (auto block = node->owningBlock(); block != defBlock;
                 block = block->owningNode()->owningBlock()) {
                addMutatedValueToBlock(mutated, block, visitedBlocks,
                                       valueToMut);
            }
        }
    }

    // Replace placeholders with real SSA values
    std::unordered_map<Value *, std::vector<Value *>> renameStacks;
    for (auto value : mutValues) renameStacks.insert({value, {}});
    renameValues(graph->block(), valueToMut, renameStacks);
}

}  // namespace jit
}  // namespace torch
