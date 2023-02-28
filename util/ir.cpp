#include "ir.h"

namespace torch {
namespace jit {

bool traversePreOrder(Block *block,
                      const std::function<bool(Node *)> &visitor) {
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
    for (auto nested : node->blocks()) rewrite(nested, pattern);
    auto newNode = pattern(node);
    if (newNode) node = newNode;
  }
}

void cloneNodesToBlock(Node *begin, Node *end, Block *block, Graph *graph,
                       std::unordered_map<Value *, Value *> &valueMap) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(begin->owningBlock() == end->owningBlock());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(begin->isBefore(end));
  for (auto iter = graph_node_list_iterator(begin, kNextDirection);
       iter != graph_node_list_iterator(end, kNextDirection); ++iter) {
    auto node = *iter;
    auto newNode = graph->createClone(
        node, [&](Value *v) { return valueMap.count(v) ? valueMap[v] : v; });
    block->appendNode(newNode);
    for (auto i = 0u; i < node->outputs().size(); i++)
      valueMap.insert({node->output(i), newNode->output(i)});
  }
}

void moveNodesToBlock(Node *begin, Node *end, Block *block, Graph *graph,
                      std::unordered_map<Value *, Value *> &valueMap) {
  cloneNodesToBlock(begin, end, block, graph, valueMap);
  graph_node_list_iterator iterBegin(end->prev(), kPrevDirection),
      iterEnd(begin->prev(), kPrevDirection);
  for (auto iter = iterBegin; iter != iterEnd; ++iter) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!(*iter)->hasUses());
    iter.destroyCurrent();
  }
}

}  // namespace jit
}  // namespace torch
