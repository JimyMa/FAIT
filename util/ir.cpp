#include "ir.h"

namespace torch {
namespace jit {

bool traverse(Block *block, const std::function<bool(Node *)> &visitor) {
    for (auto node : block->nodes()) {
        if (!visitor(node)) return false;
        for (auto nested : node->blocks())
            if (!traverse(nested, visitor)) return false;
    }
    return true;
}

void rewriteNode(Block *block, const std::function<Node *(Node *)> &pattern) {
    for (auto node = block->nodes().front(); node != block->nodes().back();
         node = node->next()) {
        auto newNode = pattern(node);
        if (newNode) node = newNode->next();
        for (auto nested : node->blocks()) rewriteNode(nested, pattern);
    }
}

}  // namespace jit
}  // namespace torch
