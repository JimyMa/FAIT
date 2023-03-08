//
// Created by jimyma on 1/27/23.
//

#ifndef LONG_TAIL_TE_FUSED_OP_H
#define LONG_TAIL_TE_FUSED_OP_H

#include "passes/te_op.h"
#include "torch/csrc/jit/ir/ir.h"

namespace torch {
namespace jit {
void FusedOpToParallization(const std::shared_ptr<Graph> &graph,
                            std::unordered_map<Value *, TypePtr> &refine_types);
}  // namespace jit
}  // namespace torch

#endif  // LONG_TAIL_TE_FUSED_OP_H
