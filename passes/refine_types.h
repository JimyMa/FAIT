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

inline TypePtr getRefinedType(Value *value, ValueTypeMap &refinedTypes) {
    if (refinedTypes.count(value))
        return refinedTypes[value];
    else
        return value->type();
}

inline void transferRefinedType(Value *src, Value *dst,
                                ValueTypeMap &refinedTypes) {
    if (!refinedTypes.count(src)) return;
    refinedTypes[dst] = refinedTypes[src];
}

void setRefinedType(Value *value, const TypePtr &newType,
                    ValueTypeMap &refinedTypes);

void removeDeadRefinedTypes(ValueTypeMap &refinedTypes, Graph *graph);

/// Lookup tables for tensor type refinement functions

extern OperatorMap<c10::SymbolicShape (*)(Node *, ValueTypeMap &)> shapeFuncs;
extern OperatorMap<c10::ScalarType (*)(Node *, ValueTypeMap &)> dtypeFuncs;
extern OperatorMap<c10::Device (*)(Node *, ValueTypeMap &)> deviceFuncs;

void initTensorTypeFuncs();

}  // namespace jit
}  // namespace torch