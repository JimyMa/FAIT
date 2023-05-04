#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace c10 {
namespace prof {

static const Symbol ns = Symbol::fromQualString("namespaces::prof");
static const Symbol Begin = Symbol::fromQualString("prof::Begin");
static const Symbol End = Symbol::fromQualString("prof::End");

}  // namespace prof
}  // namespace c10

namespace torch {
namespace jit {

namespace prof = c10::prof;

void enableProfiling();
void disableProfiling();

void profBegin(const std::string &label);
void profEnd(const std::string &label);

void printProfilingResults();

std::string fmtDuration(std::chrono::nanoseconds dur);

/// @brief Convert all `prim::Print(str label, bool begin)` to profiling
/// instrumentation.
/// @param graph The graph to be processed.
void ConvertProfilingInstrumentation(const std::shared_ptr<Graph> &graph);

}  // namespace jit
}  // namespace torch