#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

using ValueTypeMap = std::unordered_map<Value *, TypePtr>;

/// @brief Refine input types of a graph
/// @param graph The graph to be processed.
/// @param inputTypes Detailed type information of graph inputs.
/// @param refinedTypes The mappings for refined types.
void RefineInputTypes(const std::shared_ptr<Graph> &graph,
                      const std::vector<TypePtr> &inputTypes,
                      ValueTypeMap &refinedTypes);

/// @brief Infer data types and devices for tensor values.
/// @param graph The graph to be processed.
/// @param refinedTypes The mappings for refined types.
void InferDtypeAndDevice(const std::shared_ptr<Graph> &graph,
                         ValueTypeMap &refinedTypes);

/// @brief Infer shapes for tensor values.
/// @param graph The graph to be processed.
/// @param refinedTypes The mappings for refined types.
void InferShape(const std::shared_ptr<Graph> &graph,
                ValueTypeMap &refinedTypes);

/// Lookup tables for tensor type refinement functions

extern OperatorMap<c10::SymbolicShape (*)(Node *, ValueTypeMap &)> shapeFuncs;
extern OperatorMap<c10::ScalarType (*)(Node *, ValueTypeMap &)> dtypeFuncs;
extern OperatorMap<c10::Device (*)(Node *, ValueTypeMap &)> deviceFuncs;

void initTensorTypeFuncs();

}  // namespace jit
}  // namespace torch