#pragma once

#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/irange.h>

#include <chrono>
#include <iomanip>

#include "util/common.h"
#include "util/profile.h"

namespace torch {
namespace jit {

using namespace std::chrono;
using namespace std::chrono_literals;

template <class T>
inline T loadPickle(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary);
  TORCH_CHECK(ifs, "Cannot open file ", path);
  std::vector<char> buf((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
  return torch::pickle_load(buf).to<T>();
}

inline IValue processIValue(const IValue &val) {
  if (val.isList()) {
    auto list = val.toListRef();
    c10::impl::GenericList newList(list.front().type());
    for (auto &elem : list) newList.push_back(processIValue(elem));
    return std::move(newList);
  } else if (val.isTuple()) {
    auto &tuple = val.toTupleRef().elements();
    std::vector<IValue> newValues;
    for (auto &elem : tuple) newValues.push_back(processIValue(elem));
    return c10::ivalue::Tuple::create(std::move(newValues));
  } else if (val.isTensor()) {
    return val.toTensor().cuda();
  } else
    return val;
}

inline Stack getFeatureSample(const c10::List<IValue> &dataset, size_t index) {
  auto tup = dataset.get(index).toTupleRef().elements();
  Stack inputs;
  inputs.push_back({});
  for (auto &val : tup) inputs.push_back(processIValue(val));
  return std::move(inputs);
}

static constexpr auto kWarmupRuns = 16;
static constexpr auto kRunDuration = 2s;

inline auto evaluate(const std::function<void(size_t)> &task) {
  // Warm up
  for (auto i : c10::irange(kWarmupRuns)) task(i);
  at::cuda::device_synchronize();

  // Run for the expected period
  size_t count = 0;
  auto begin = system_clock::now();
  while (system_clock::now() - begin < kRunDuration) {
    task(count++);
    at::cuda::device_synchronize();
  }

  return (system_clock::now() - begin) / int64_t(count);
}

}  // namespace jit
}  // namespace torch