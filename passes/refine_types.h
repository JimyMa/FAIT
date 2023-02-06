#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

using ValueTypeMap = std::unordered_map<Value *, TypePtr>;

void RefineInputTypes(const std::shared_ptr<Graph> &graph,
                      const std::vector<TypePtr> &inputTypes,
                      ValueTypeMap &refinedTypes);

void InferDtypeAndDevice(const std::shared_ptr<Graph> &graph,
                         ValueTypeMap &refinedTypes);

/// Lookup tables for tensor type refinement functions

extern OperatorMap<std::vector<c10::ScalarType> (*)(Node *)> dtypeFuncs;
extern OperatorMap<std::vector<c10::Device> (*)(Node *)> deviceFuncs;
extern OperatorMap<std::vector<c10::SymbolicShape> (*)(Node *)> shapeFuncs;

}  // namespace jit
}  // namespace torch