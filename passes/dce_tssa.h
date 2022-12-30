#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

/// @brief Eliminate dead code when TensorSSA operations are present.
/// @param graph The graph to be optimized.
void EliminateDeadCodeTSSA(const std::shared_ptr<Graph> &graph);

}  // namespace jit
}  // namespace torch