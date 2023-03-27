#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/ops/allclose.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/serialize.h>
#include <torchvision/vision.h>

#include <chrono>
#include <iomanip>

#include "passes/canonicalize.h"
#include "passes/common_passes.h"
#include "passes/freeze_module.h"
#include "passes/fuse_ops.h"
#include "passes/parallelize_loops.h"
#include "passes/refine_types.h"
#include "passes/te_op.h"
#include "passes/tensor_ssa.h"
#include "passes/type_utils.h"
#include "passes/validate_graph.h"
#include "util/logging.h"
#include "util/rand.h"

using namespace torch::jit;
using namespace std::chrono;
using namespace std::chrono_literals;

static void dumpGraphToFile(const std::shared_ptr<Graph> &graph,
                            const std::string &path) {
  std::ofstream ofs(path);
  graph->print(ofs);
}

static IValue genValueOfType(const TypePtr &type) {
  switch (type->kind()) {
    case TypeKind::TensorType:
      return generateRandomTensor(type->cast<TensorType>());

    case TypeKind::TupleType: {
      auto elements = type->cast<TupleType>()->elements();
      c10::impl::GenericList list(getUnifiedElementType(type));
      for (auto &elem : elements) list.push_back(genValueOfType(elem));
      return list;
    }

    default:
      TORCH_CHECK(false, "Cannot generate input value for type ", *type);
  }
}

static Stack generateRandomInputs(const std::vector<TypePtr> &inputTypes) {
  Stack inputs;
  inputs.push_back({});
  for (auto &type : inputTypes) inputs.push_back(genValueOfType(type));
  return inputs;
}

static std::string fmtIndices(const std::vector<size_t> &indices) {
  std::stringstream ss;
  print(ss, '[', indices.front(), ']');
  if (indices.size() == 1) return ss.str();
  ss << '[';
  for (auto i : c10::irange(1, indices.size())) {
    if (i > 1) ss << ", ";
    ss << indices[i];
  }
  ss << ']';
  return ss.str();
}

static void checkValue(const IValue &real, const IValue &ref,
                       std::vector<size_t> &indices) {
  TORCH_CHECK(real.tagKind() == ref.tagKind(), "Expect ", ref.tagKind(),
              ", got ", real.tagKind(), " at ", fmtIndices(indices));
  if (real.isTensor()) {
    TORCH_CHECK(at::allclose(real.toTensor(), ref.toTensor(), 1e-3, 1e-5),
                "Inconsistent tensor value at ", fmtIndices(indices));
  } else if (real.isList()) {
    auto realList = real.toListRef(), refList = ref.toListRef();
    TORCH_CHECK(realList.size() == refList.size(), "Expect list of length ",
                refList.size(), ", got ", realList.size(), " at ",
                fmtIndices(indices));
    for (auto i : c10::irange(realList.size())) {
      indices.push_back(i);
      checkValue(realList[i], refList[i], indices);
      indices.pop_back();
    }
  } else if (real.isTuple()) {
    auto &realTup = real.toTupleRef().elements(),
         &refTup = ref.toTupleRef().elements();
    TORCH_CHECK(realTup.size() == refTup.size(), "Expect tuple of length ",
                refTup.size(), ", got ", realTup.size(), " at ",
                fmtIndices(indices));
    for (auto i : c10::irange(realTup.size())) {
      indices.push_back(i);
      checkValue(realTup[i], refTup[i], indices);
      indices.pop_back();
    }
  } else {
    TORCH_CHECK(real == ref, "Unequal value at ", fmtIndices(indices));
  }
}

static void checkOutputs(const Stack &realOutputs, const Stack &refOutputs) {
  TORCH_CHECK(realOutputs.size() == refOutputs.size());
  std::vector<size_t> indices;
  for (auto i : c10::irange(refOutputs.size())) {
    auto &realVal = realOutputs[i], &refVal = refOutputs[i];
    indices.push_back(i);
    checkValue(realVal, refVal, indices);
    indices.pop_back();
  }
}

static constexpr auto kWarmupRuns = 3;
static constexpr auto kRunDuration = 2s;

static auto evaluate(const std::function<void(size_t)> &task) {
  // Warm up
  for (auto _ : c10::irange(kWarmupRuns)) task(0);
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

static std::string fmtDuration(std::chrono::duration<size_t, std::nano> dur) {
  double fp = dur.count();
  auto unitIdx = 0;
  while (unitIdx < units.size() - 1 && fp > 1e3) {
    fp /= 1e3;
    unitIdx++;
  }
  std::stringstream ss;
  print(ss, std::setprecision(4), fp, units[unitIdx]);
  return ss.str();
}

static void dumpStruct(const IValue &val, size_t indent = 0) {
  for (auto _ : c10::irange(indent)) std::cout << "  ";
  if (val.isList()) {
    std::cout << "List\n";
    for (auto &elem : val.toListRef()) dumpStruct(elem, indent + 1);
  } else {
    std::cout << *val.type() << '\n';
  }
}

static IValue processIValue(const IValue &val) {
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
  } else
    return val;
}

static Stack getSample(const c10::List<IValue> &dataset, size_t index) {
  auto tup = dataset.get(index).toTupleRef().elements();
  Stack inputs;
  inputs.push_back({});
  for (auto &val : tup) inputs.push_back(processIValue(val));
  return std::move(inputs);
}

int main(int argc, const char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage: example <script-module> <input-types> <input-data>?\n";
    return 1;
  }
  at::globalContext().lazyInitCUDA();
  Module mod;
  try {
    mod = load(argv[1]);
  } catch (c10::Error &e) {
    std::cerr << e.what();
    return 1;
  } catch (ErrorReport &e) {
    std::cerr << e.what();
    return 1;
  }
  Freeze(&mod);
  auto graph = mod.get_method("forward").graph();
  auto origin_graph = graph->copy();
  auto inputTypes = parseInputTypes(argv[2]);
  ValueTypeMap refinedTypes;
  try {
    RefineInputTypes(graph, inputTypes, refinedTypes);
    CanonicalizeOps(graph);
    ToTensorSSA(graph);
    dumpGraphToFile(graph, "after_tssa.rb");
    ParallelizeLoops(graph);
    InferDtypeAndDevice(graph, refinedTypes);
    InferShape(graph, refinedTypes);
    dumpGraphToFile(graph, "after_par.rb");
    FuseOps(graph, refinedTypes);
    dumpGraphToFile(graph, "after_fuse.rb");
    SplitParallelMaps(graph, refinedTypes);
    dumpGraphToFile(graph, "after_split.rb");
    ToMutableTensors(graph);
    ConvertInfusibleMapsToLoops(graph, refinedTypes);
    CanonicalizeFusableMaps(graph);
    dumpGraphToFile(graph, "after_back.rb");
    MapFunctorToParallization(graph, refinedTypes);
    FusedOpToParallization(graph, refinedTypes);
    dumpGraphToFile(graph, "after_codegen.rb");
    Validate(graph);
  } catch (c10::Error &err) {
    std::cout << err.what();
    dumpGraphToFile(graph, "error.rb");
  } catch (ErrorReport &err) {
    std::cout << err.what();
    dumpGraphToFile(graph, "error.rb");
  }

  // Runtime
  c10::impl::GenericList dataset(AnyType::get());
  size_t numSamples = 0;
  if (argc > 3) {
    std::ifstream ifs(argv[3], std::ios::binary);
    TORCH_CHECK(ifs);
    std::vector<char> buf((std::istreambuf_iterator<char>(ifs)),
                          std::istreambuf_iterator<char>());
    dataset = torch::pickle_load(buf).toList();
    numSamples = dataset.size();
  } else {
    dataset.emplace_back(
        c10::ivalue::Tuple::create(generateRandomInputs(inputTypes)));
    numSamples = 1;
  }

  Stack stack;

  Code code(graph, "");
  stack = getSample(dataset, 0);
  LONG_TAIL_LOG_INFO("RUN LONG TAIL BEGIN");
  torch::jit::InterpreterState(code).run(stack);
  LONG_TAIL_LOG_INFO("RUN LONG TAIL DONE");
  auto output_tss_parallel = stack;

  GraphFunction origin_function("original", origin_graph, nullptr);
  stack = getSample(dataset, 0);
  LONG_TAIL_LOG_INFO("RUN TS BEGIN");
  origin_function.run(stack);
  LONG_TAIL_LOG_INFO("RUN TS DONE");
  auto output_origin = stack;

  checkOutputs(output_tss_parallel, output_origin);

  {
    auto dur = evaluate([&](size_t i) {
      auto stack = getSample(dataset, i % numSamples);
      torch::jit::InterpreterState(code).run(stack);
    });
    print(std::cout, "long tail latency: ", fmtDuration(dur), '\n');
  }
  {
    auto dur = evaluate([&](size_t i) {
      auto stack = getSample(dataset, i % numSamples);
      origin_function.run(stack);
    });
    print(std::cout, "ts latency: ", fmtDuration(dur), '\n');
  }
}
