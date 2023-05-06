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
static constexpr auto kStatWidth = 10;

static void printLabel(const std::string &label) {
  print(std::cout, std::setw(kLabelWidth), std::setiosflags(std::ios::left),
        label, std::resetiosflags(std::ios::left));
}

template <class T>
static void printStat(T &&stat) {
  print(std::cout, std::setw(kStatWidth), stat);
}

void printProfilingResults() {
  if (records.empty()) return;

  // Items
  std::cout << "\nProfiling results:\n";
  printLabel("Label");
  printStat("Count");
  printStat("Total");
  printStat("Mean");
  printStat("Min");
  printStat("Max");
  std::cout << '\n';

  for (auto &label : labels) {
    auto &record = records[label];
    printLabel(label);
    printStat(record.count);  // count
    if (record.count == 0) {
      std::cout << '\n';
      continue;
    }
    printStat(fmtDuration(record.total));                 // total
    printStat(fmtDuration(record.total / record.count));  // mean
    printStat(fmtDuration(record.min));                   // min
    printStat(fmtDuration(record.max));                   // max
    std::cout << '\n';
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