#include "profile.h"

#include <c10/cuda/CUDAFunctions.h>

#include <chrono>

#include "util/common.h"
#include "util/ir.h"

namespace torch {
namespace jit {

using namespace std::chrono;

static bool enabled = false;

void enableProfiling() { enabled = true; }
void disableProfiling() { enabled = false; }

struct TimeRecord {
  nanoseconds total{0}, min{INT64_MAX}, max{0};
  size_t count = 0;
  c10::optional<system_clock::time_point> begin = c10::nullopt;
};

static std::vector<std::string> labels;
static std::unordered_map<std::string, TimeRecord> records;

void profBegin(const std::string &label) {
  if (!enabled) return;
  at::cuda::device_synchronize();
  if (!records.count(label)) {
    labels.push_back(label);
    records.insert({label, {}});
  }
  records[label].begin = system_clock::now();
}

void profEnd(const std::string &label) {
  if (!enabled) return;
  at::cuda::device_synchronize();
  auto &record = records.at(label);
  TORCH_CHECK(record.begin.has_value(),
              "`beginProfile` has not been called before.");
  auto dur = system_clock::now() - *record.begin;
  record.begin = c10::nullopt;
  record.count++;
  record.total += dur;
  record.min = std::min(record.min, dur);
  record.max = std::max(record.max, dur);
}

static std::array<std::string, 4> units{"ns", "us", "ms", "s"};

std::string fmtDuration(nanoseconds dur) {
  double fp = dur.count();
  auto unitIdx = 0;
  while (unitIdx < units.size() - 1 && fp > 1e3) {
    fp /= 1e3;
    unitIdx++;
  }
  std::stringstream ss;
  ss << std::setprecision(4) << fp << units[unitIdx];
  return ss.str();
}

static constexpr auto kLabelWidth = 16;
static constexpr auto kTimeWidth = 8;

void printProfilingResults() {
  if (records.empty()) return;
  print(std::cout, "\nProfiling results:\n");
  print(std::cout, std::setw(kLabelWidth), std::setiosflags(std::ios::left),
        "Label", std::resetiosflags(std::ios::left), std::setw(kTimeWidth),
        "Count", std::setw(kTimeWidth), "Mean", std::setw(kTimeWidth), "Min",
        std::setw(kTimeWidth), "Max", '\n');
  for (auto &label : labels) {
    auto &record = records[label];
    print(std::cout, std::setw(kLabelWidth), std::setiosflags(std::ios::left),
          label, std::resetiosflags(std::ios::left));
    print(std::cout, std::setw(kTimeWidth), record.count);
    if (record.count == 0) {
      std::cout << '\n';
      continue;
    }
    print(std::cout, std::setw(kTimeWidth),
          fmtDuration(record.total / record.count), std::setw(kTimeWidth),
          fmtDuration(record.min), std::setw(kTimeWidth),
          fmtDuration(record.max), '\n');
  }
}

static auto _registry = RegisterOperators()
                            .op("prof::Begin(str label) -> ()", profBegin,
                                RegisterOperators::options().aliasAnalysis(
                                    c10::AliasAnalysisKind::CONSERVATIVE))
                            .op("prof::End(str label) -> ()", profEnd,
                                RegisterOperators::options().aliasAnalysis(
                                    c10::AliasAnalysisKind::CONSERVATIVE));

void ConvertProfilingInstrumentation(const std::shared_ptr<Graph> &graph) {
  rewrite(graph->block(), [&](Node *node) -> Node * {
    if (node->kind() != prim::Print) return nullptr;
    if (node->inputs().size() != 2) return nullptr;
    auto label = constant_as<std::string>(node->input(0));
    auto begin = constant_as<bool>(node->input(1));
    if (!label.has_value() || !begin.has_value()) return nullptr;
    auto newNode = *begin ? graph->create(prof::Begin, {node->input(0)}, 0)
                          : graph->create(prof::End, {node->input(0)}, 0);
    TORCH_CHECK(newNode->maybeOperator());
    return replace(node, newNode);
  });
}

}  // namespace jit
}  // namespace torch