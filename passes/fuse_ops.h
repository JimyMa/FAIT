#pragma once

#include <torch/csrc/jit/ir/ir.h>

#include "refine_types.h"

namespace torch {
namespace jit {

extern OperatorSet fusableOps;

void FuseOps(const std::shared_ptr<Graph> &graph, ValueTypeMap &refinedTypes);

void printOpsInFusionGroups(const std::shared_ptr<Graph> &graph);

Node *commitFusion(Node *head, Node *tail, Graph *graph,
                   ValueTypeMap &refinedTypes);

}  // namespace jit
}  // namespace torch
