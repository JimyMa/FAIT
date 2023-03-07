#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace c10 {
namespace tssa {

static const Symbol ns = Symbol::fromQualString("namespaces::tssa");
static const Symbol Assign = Symbol::fromQualString("tssa::Assign");
static const Symbol Update = Symbol::fromQualString("tssa::Update");

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

void ToTensorSSA(const std::shared_ptr<Graph> &graph);

}  // namespace jit
}  // namespace torch
