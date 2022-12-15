#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

bool containsLoop(Block *block);

std::unordered_map<Value *, Node *> findParentLoopOfValues(Block *block);

}  // namespace jit
}  // namespace torch
