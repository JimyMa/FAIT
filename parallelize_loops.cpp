#include "parallelize_loops.h"

#include "loop_analysis.h"
#include "util.h"

namespace c10 {
namespace prim {

Symbol ParallelLoop = Symbol::prim("ParallelLoop");

}
}  // namespace c10

namespace torch {
namespace jit {

void ParallelizeLoops(const std::shared_ptr<Graph> &graph) {
    // Find all innermost loops
    std::vector<Node *> loops;
    walk(graph->block(), [&](Node *node) {
        if (node->kind() != prim::Loop) return true;
        auto block = node->blocks()[0];
        if (containsLoop(block)) return true;  // skip nested loop
        loops.push_back(node);
        return true;
    });

    //

    for (auto node : loops) {
        node->dump();
    }

    return;
}

}  // namespace jit
}  // namespace torch
