#include "ir.h"

namespace torch {
namespace jit {

bool traversePreOrder(Block *block, const std::function<bool(Node *)> &visitor) {
    for (auto node : block->nodes()) {
        if (!visitor(node)) return false;
        for (auto nested : node->blocks())
            if (!traversePreOrder(nested, visitor)) return false;
    }
    return true;
}

bool traversePostOrder(Block *block,
                       const std::function<bool(Node *)> &visitor) {
    for (auto node : block->nodes()) {
        for (auto nested : node->blocks())
            if (!traversePostOrder(nested, visitor)) return false;
        if (!visitor(node)) return false;
    }
    return true;
}

void rewrite(Block *block, const std::function<Node *(Node *)> &pattern) {
    for (auto node = block->nodes().front(); node != block->nodes().back();
         node = node->next()) {
        auto newNode = pattern(node);
        if (newNode) node = newNode;
        for (auto nested : node->blocks()) rewrite(nested, pattern);
    }
}

}  // namespace jit
}  // namespace torch
