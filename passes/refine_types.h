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

extern OperatorMap<c10::SymbolicShape (*)(Node *, ValueTypeMap &)> shapeFuncs;
extern OperatorMap<c10::ScalarType (*)(Node *, ValueTypeMap &)> dtypeFuncs;
extern OperatorMap<c10::Device (*)(Node *, ValueTypeMap &)> deviceFuncs;

void initTensorTypeFuncs();

}  // namespace jit
}  // namespace torch