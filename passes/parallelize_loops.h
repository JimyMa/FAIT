#pragma once

#include <torch/csrc/jit/ir/ir.h>

#include "refine_types.h"
#include "util/common.h"

namespace c10 {
namespace prim {

static auto ParallelMap = Symbol::prim("ParallelMap");

}
}  // namespace c10

namespace torch {
namespace jit {

/// @brief Convert parallelizable loops in the graph to `ParallelMap`s.
/// @param graph The graph to be parallelized.
void ParallelizeLoops(const std::shared_ptr<Graph> &graph);

/// @brief Split `ParallelMap`s to several parts such that each `FusionGroup` is
/// in its exclusive `ParallelMap`.
/// @param graph The graph to be processed.
void SplitParallelMaps(const std::shared_ptr<Graph> &graph,
                       ValueTypeMap &refinedTypes);

/// @brief Convert `ParallelMap`s without a `FusionGroup` to normal for-loops.
/// @param graph The graph to be processed.
void ConvertInfusibleMapsToLoops(const std::shared_ptr<Graph> &graph,
                                 ValueTypeMap &refinedTypes);

/// @brief Reorder block parameters of `FusionGroup` inside `ParallelMap` such
/// that they are consistent with the ones of `ParallelMap`.
/// @param graph The graph to be processed.
void CanonicalizeFusableMaps(const std::shared_ptr<Graph> &graph);

inline c10::optional<size_t> getParMapTripCount(Node *parMap) {
  auto lenIVal = toIValue(parMap->input(0));
  return mapOpt<size_t>(
      lenIVal, [](const IValue &ival) { return size_t(ival.toInt()); });
}

}  // namespace jit
}  // namespace torch
