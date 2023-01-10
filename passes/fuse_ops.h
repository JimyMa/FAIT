#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

extern OperatorSet fusableOps;

void FuseOps(const std::shared_ptr<Graph> &graph);

}  // namespace jit
}  // namespace torch
