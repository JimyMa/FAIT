#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace c10 {
namespace prim {

extern Symbol ParallelMap;

}
}  // namespace c10

namespace torch {
namespace jit {

void ParallelizeLoops(const std::shared_ptr<Graph> &graph);

}  // namespace jit
}  // namespace torch
