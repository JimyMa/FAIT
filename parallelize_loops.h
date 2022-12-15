#pragma once

#include "loop_analysis.h"

namespace c10 {
namespace prim {

extern Symbol ParallelLoop;

}
}

namespace torch {
namespace jit {

void ParallelizeLoops(const std::shared_ptr<Graph> &graph);

}  // namespace jit
}  // namespace torch
