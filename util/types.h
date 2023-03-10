#pragma once

#include <c10/core/ScalarType.h>

namespace torch {
namespace jit {

/* Scalar types */

template <class T>
struct GetScalarType {
  static constexpr auto result = c10::ScalarType::NumOptions;
};

template <>
struct GetScalarType<bool> {
  static constexpr auto result = c10::ScalarType::Bool;
};

template <>
struct GetScalarType<int64_t> {
  static constexpr auto result = c10::ScalarType::Long;
};

template <>
struct GetScalarType<float> {
  static constexpr auto result = c10::ScalarType::Float;
};

}  // namespace jit
}  // namespace torch
