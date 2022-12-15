#include "loop_analysis.h"

#include "util.h"

namespace torch {
namespace jit {

bool containsLoop(Block *block) {
    bool result = false;
    walk(block, [&](Node *node) {
        if (node->kind() == prim::Loop) {
            result = true;
            return false;
        }
        return true;
    });
    return result;
}

static void collectValue(Block *block, Node *loop,
                         std::unordered_map<Value *, Node *> &valueToLoop) {
    for (auto value : block->inputs()) valueToLoop.insert({value, loop});
    for (auto node : block->nodes()) {
        for (auto value : node->outputs()) valueToLoop.insert({value, loop});
        auto newLoop = loop;
        if (node->kind() == prim::Loop) newLoop = node;
        for (auto nested : node->blocks())
            collectValue(nested, newLoop, valueToLoop);
    }
}

std::unordered_map<Value *, Node *> findParentLoopOfValues(Block *block) {
    std::unordered_map<Value *, Node *> valueToLoop;
    collectValue(block, nullptr, valueToLoop);
    return valueToLoop;
}

}  // namespace jit
}  // namespace torch
