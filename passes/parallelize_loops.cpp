#include "parallelize_loops.h"

#include "util/ir.h"

namespace c10 {
namespace prim {

Symbol ParallelLoop = Symbol::prim("ParallelLoop");

}
}  // namespace c10

namespace torch {
namespace jit {

inline static bool containsLoop(Block *block) {
    bool result = false;
    traverse(block, [&](Node *node) {
        if (node->kind() == prim::Loop) {
            result = true;
            return false;
        }
        return true;
    });
    return result;
}

void ParallelizeLoops(const std::shared_ptr<Graph> &graph) {
    // Find all innermost loops
    std::vector<Node *> loops;
    traverse(graph->block(), [&](Node *node) {
        if (node->kind() != prim::Loop) return true;
        auto block = node->blocks()[0];
        if (containsLoop(block)) return true;  // skip nested loop
        loops.push_back(node);
        return true;
    });

    //

    return;
}

}  // namespace jit
}  // namespace torch
