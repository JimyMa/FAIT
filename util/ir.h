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
/// post-order. Note that the actual rewrite should be done by the user, and
/// this function will NOT mutates the IR.
/// @param block The block to be rewritten.
/// @param pattern The rewrite pattern, which returns a new node if the rewrite
/// is successfully applied and the following traversal should begin right after
/// this node, and nullptr otherwise.
void rewrite(Block *block, const std::function<Node *(Node *)> &pattern);

/// @brief Remove the node, and return the one right before it.
/// @param node The node to be removed.
/// @return The node right before the removed one.
inline Node *remove(Node *node) {
    auto prev = node->prev();
    node->destroy();
    return prev;
}

/// @brief Clone the nodes in range [`begin`, `end`) to the end of the new
/// block.
/// @param begin The beginning of the node range (inclusive).
/// @param end The end of the node range (exclusive). Must be in the same block
/// as `begin` and be after `begin` topologically.
/// @param block The new block to clone nodes to.
/// @param graph The graph that owns the block.
/// @param valueMap Mappings from the value in the original block to the ones in
/// the new block.
void cloneNodesToBlock(Node *begin, Node *end, Block *block, Graph *graph,
                       std::unordered_map<Value *, Value *> &valueMap);

/// @brief Move the nodes in range [`begin`, `end`) to the end of the new block.
/// @param begin The beginning of the node range (inclusive).
/// @param end The end of the node range (exclusive). Must be in the same block
/// as `begin` and be after `begin` topologically.
/// @param block The new block to clone nodes to.
/// @param graph The graph that owns the block.
/// @param valueMap Mappings from the value in the original block to the ones in
/// the new block.
void moveNodesToBlock(Node *begin, Node *end, Block *block, Graph *graph,
                      std::unordered_map<Value *, Value *> &valueMap);

}  // namespace jit
}  // namespace torch
