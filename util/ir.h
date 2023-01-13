#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

/// @brief Traverse the nodes in the block in pre-order (node first, then its
/// nested blocks).
/// @param block The block to be traversed.
/// @param visitor Visitor function. Returns true if the traversal continues,
/// and aborts otherwise.
/// @return If the traversal terminates without abortion.
bool traversePreOrder(Block *block, const std::function<bool(Node *)> &visitor);

/// @brief Traverse the nodes in the block in pre-order (node first, then its
/// nested blocks).
/// @param block The block to be traversed.
/// @param visitor Visitor function. Returns true if the traversal continues,
/// and aborts otherwise.
/// @return If the traversal terminates without abortion.
bool traversePostOrder(Block *block,
                       const std::function<bool(Node *)> &visitor);

/// @brief Replace the node with a new node.
/// @param oldNode The old node to be replaced.
/// @param newNode The new node that will replace the old one.
inline void replace(Node *oldNode, Node *newNode) {
    newNode->insertAfter(oldNode);
    oldNode->replaceAllUsesWith(newNode);
    oldNode->destroy();
}

/// @brief Rewrite nodes in a block with a given pattern recursively in
/// pre-order. Note that the actual rewrite should be done by the user, and this
/// function will NOT mutates the IR.
/// @param block The block to be rewritten.
/// @param pattern The rewrite pattern, which returns a new node if the rewrite
/// is successfully applied and the following traversal should begin right after
/// this node, and nullptr otherwise.
void rewrite(Block *block, const std::function<Node *(Node *)> &pattern);

}  // namespace jit
}  // namespace torch
