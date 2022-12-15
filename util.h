#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

bool walk(Block *block, const std::function<bool(Node *)> &visitor);

void rewriteNode(Block *block, const std::function<Node *(Node *)> &pattern);

}  // namespace jit
}  // namespace torch
