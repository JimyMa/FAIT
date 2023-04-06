#pragma once

#include "ir_utils.h"

namespace torch {
namespace jit {
namespace tensorexpr {

/// @brief Collect all intermediate buffers from a statement.
/// @param stmt The statement to be analyzed.
/// @param outBufs Buffers that are known to be outputs of the statement.
/// @return A vector of unique intermediate buffers.
std::vector<BufPtr> collectIntermBufs(
    const StmtPtr &stmt, const std::unordered_set<BufPtr> &outBufs);

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch