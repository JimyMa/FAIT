#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace c10 {
namespace tssa {

extern Symbol ns;
extern Symbol Assign, Update;

}  // namespace tssa
}  // namespace c10

namespace torch {
namespace jit {

inline Node *createTssaAssign(Graph *graph, Value *dst, Value *src) {
    return graph->create(c10::tssa::Assign, {dst, src});
}

inline Node *createTssaUpdate(Graph *graph, Value *tensor, Value *cause) {
    return graph->create(c10::tssa::Update, {tensor, cause});
}

void ToTensorSSA(const std::shared_ptr<Graph> &graph);

}  // namespace jit
}  // namespace torch
