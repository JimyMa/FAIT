//
// Created by jimyma on 1/27/23.
//

#ifndef LONG_TAIL_TE_OP_H
#define LONG_TAIL_TE_OP_H

#include "torch/csrc/jit/ir/ir.h"

namespace c10 {
namespace tssa {

static auto ParallelledFunctor = Symbol::fromQualString("tssa::ParallelledFunctor");
static auto parallel_degree = Symbol::fromQualString("attr::parallel_degree");
static auto is_parallelled_args = Symbol::fromQualString("attr::is_parallelled_args");
static auto input_refine_types = Symbol::fromQualString("attr::input_refine_types");

}  // namespace tssa
}  // namespace c10

namespace torch {
namespace jit {
void MapFunctorToParallization(const std::shared_ptr<Graph> &graph,
                               std::unordered_map<Value *, TypePtr>& refine_types);
}  // namespace torch
}  // namespace jit


#endif //LONG_TAIL_TE_OP_H
