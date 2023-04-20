#pragma once

#include <c10/cuda/CUDAFunctions.h>

#include <chrono>
#include <iomanip>

#include "util/common.h"

namespace torch {
namespace jit {

using namespace std::chrono;
using namespace std::chrono_literals;

static constexpr auto kWarmupRuns = 16;
static constexpr auto kRunDuration = 2s;

inline auto evaluate(const std::function<void(size_t)> &task) {
  // Warm up
  for (auto i : c10::irange(kWarmupRuns)) task(i);
  cuda::device_synchronize();

  // Run for the expected period
  size_t count = 0;
  auto begin = system_clock::now();
  while (system_clock::now() - begin < kRunDuration) {
    task(count++);
    cuda::device_synchronize();
  }

  return (system_clock::now() - begin) / count;
}

static std::array<std::string, 4> units{"ns", "us", "ms", "s"};

inline std::string fmtDuration(std::chrono::duration<size_t, std::nano> dur) {
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
}  // namespace jit
}  // namespace torch