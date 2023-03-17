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
#include <time.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/serialize.h>
#include <torchvision/vision.h>

#include "fuser/tssa_set_ops.h"
#include "passes/canonicalize.h"
#include "passes/common_passes.h"
#include "passes/freeze_module.h"
#include "passes/fuse_ops.h"
#include "passes/parallelize_loops.h"
#include "passes/refine_types.h"
#include "passes/te_op.h"
#include "passes/tensor_ssa.h"
#include "passes/validate_graph.h"
#include "util/logging.h"

using namespace torch::jit;

static void dumpGraphToFile(const std::shared_ptr<Graph> &graph,
                            const std::string &path) {
  std::ofstream ofs(path);
  graph->print(ofs);
}

int main(int argc, const char *argv[]) {
  at::globalContext().lazyInitCUDA();
  if (argc < 2) {
    std::cerr << "usage: example <path-to-script-module>\n";
    return 1;
  }
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
  // std::vector<TypePtr> inputTypes{TupleType::create({
  //     TensorType::createContiguous(c10::kFloat, c10::kCUDA, {1, 255, 10,
  //     10}), TensorType::createContiguous(c10::kFloat, c10::kCUDA, {1, 255,
  //     20, 20}), TensorType::createContiguous(c10::kFloat, c10::kCUDA, {1,
  //     255, 40, 40}),
  // })};
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
  at::List<at::Tensor> a_list = {
      at::rand({1, 255, 10, 10}).to(at::kFloat).cuda(),
      at::rand({1, 255, 20, 20}).to(at::kFloat).cuda(),
      at::rand({1, 255, 40, 40}).to(at::kFloat).cuda()};
  Code code(graph, "");
  Stack input = {"", a_list};

  LONG_TAIL_LOG_INFO("RUN LONG TAIL BEGIN");
  torch::jit::InterpreterState(code).run(input);
  LONG_TAIL_LOG_INFO("RUN LONG TAIL DONE");
  auto output_tss_parallel = input[0].toTensorList();

  GraphFunction origin_function("simple_simple_loop", origin_graph, nullptr);
  input = {"", a_list};
  LONG_TAIL_LOG_INFO("RUN TS BEGIN");
  origin_function.run(input);
  LONG_TAIL_LOG_INFO("RUN TS DONE");

  input = {"", a_list};
  Code ts_code(origin_graph, "");
  LONG_TAIL_LOG_INFO("RUN TS BEGIN");
  torch::jit::InterpreterState(ts_code).run(input);
  LONG_TAIL_LOG_INFO("RUN TS DONE");
  auto output_origin = input[0].toTensorList();

  std::cout << "Checking Pass: "
            << at::allclose(output_tss_parallel[0], output_origin[0])
            << std::endl;

  time_t start, mid, end;

  start = clock();
  for (int i = 0; i < 10000; i++) {
    input = {"", a_list};
    torch::jit::InterpreterState(code).run(input);
  }
  mid = clock();
  std::cout << "long tail time: " << (mid - start) / 1000 << "ms" << std::endl;
  mid = clock();
  for (int i = 0; i < 10000; i++) {
    // GraphFunction origin_function("normalize", origin_graph, nullptr);
    input = {"", a_list};
    origin_function.run(input);
  }
  end = clock();

  std::cout << "ts time: " << (end - mid) / 1000 << "ms" << std::endl;
}
