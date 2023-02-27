#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace c10 {
namespace tssa {

static auto ns = Symbol::fromQualString("namespaces::tssa");
static auto Assign = Symbol::fromQualString("tssa::Assign");
static auto Update = Symbol::fromQualString("tssa::Update");

}  // namespace tssa
}  // namespace c10

namespace torch {
namespace jit {

namespace tssa = c10::tssa;

inline Node *createTssaAssign(Graph *graph, Value *dst, Value *src) {
    return graph->create(tssa::Assign, {dst, src});
}

inline Node *createTssaUpdate(Graph *graph, Value *tensor, Value *cause) {
    return graph->create(tssa::Update, {tensor, cause});
}

/// @brief Convert a graph to TensorSSA form. All mutations of tensors will be
/// eliminated.
/// @param graph The graph to be transformed.
void ToTensorSSA(const std::shared_ptr<Graph> &graph);

/// @brief Convert a graph out of TensorSSA form such that it can be directly
/// executed by the TorchScript interpreter. Note that all TensorSSA operations
/// in `FusionGroup` will be kept.
/// @param graph The graph to be transformed.
void ToMutableTensors(const std::shared_ptr<Graph> &graph);

}  // namespace jit
}  // namespace torch
