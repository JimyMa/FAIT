#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

inline bool isAliasing(Node *node) {
  auto schema = node->maybeSchema();
  if (!schema) return false;
  if (schema->arguments().empty()) return false;
  return schema->is_aliasing({c10::SchemaArgType::input, 0});
}

inline bool isMutating(Node *node) {
  auto schema = node->maybeSchema();
  if (!schema) return false;
  if (schema->arguments().empty()) return false;
  return schema->is_mutable({c10::SchemaArgType::input, 0});
}

inline bool isMutated(Value *value) {
  auto &uses = value->uses();
  return std::any_of(uses.begin(), uses.end(), [](const Use &use) {
    return use.offset == 0 && isMutating(use.user);
  });
}

}  // namespace jit
}  // namespace torch
