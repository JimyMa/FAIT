#include <torch/csrc/jit/ir/node_hashing.h>

#include "common_passes.h"
#include "util/ir.h"
#include "util/traits.h"

namespace torch {
namespace jit {

static bool mayConsider(Node *node) {
    // Skip mutating nodes
    if (isMutating(node)) return false;

    // Skip nodes with blocks
    if (!node->blocks().empty()) return false;

    // Skip nodes with mutated inputs or outputs
    for (auto input : node->inputs()) {
        if (isMutated(input)) return false;
    }
    for (auto output : node->outputs()) {
        if (isMutated(output)) return false;
    }

    return true;
}

void EliminateCommonSubexprTSSA(const std::shared_ptr<Graph> &graph) {
    std::unordered_set<Node *, HashNode, EqualNode> subexprs;
    rewrite(graph->block(), [&](Node *node) -> Node * {
        if (!mayConsider(node)) return nullptr;
        auto iter = subexprs.find(node);
        if (iter != subexprs.end()) {
            auto existing = *iter;
            node->replaceAllUsesWith(existing);
            return remove(node);
        } else {
            subexprs.insert(node);
            return nullptr;
        }
    });
}

}  // namespace jit
}  // namespace torch
