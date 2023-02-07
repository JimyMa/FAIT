//
// Created by jimyma on 1/27/23.
//

#ifndef LONG_TAIL_TE_OP_H
#define LONG_TAIL_TE_OP_H

#include "torch/csrc/jit/ir/ir.h"

using namespace torch::jit;
namespace torch {
namespace jit {
void MapFunctorToParallization(const std::shared_ptr<Graph> &graph,
                               const std::unordered_map<Value *, TypePtr>& refine_types);
}  // namespace torch
}  // namespace jit


#endif //LONG_TAIL_TE_OP_H
