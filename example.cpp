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
#include <torchvision/vision.h>

#include "passes/common_passes.h"
#include "passes/freeze_module.h"
#include "passes/fuse_ops.h"
#include "passes/parallelize_loops.h"
#include "passes/refine_types.h"
#include "passes/te_fused_op.h"
#include "passes/te_op.h"
#include "passes/tensor_ssa.h"
#include "passes/validate_graph.h"

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
  auto graph = mod.get_method("forward").graph();
  auto origin_graph = graph->copy();
  std::vector<TypePtr> inputTypes{
      TensorType::createContiguous(c10::kFloat, c10::kCUDA, {1, 85, 1, 1})};
  ValueTypeMap refinedTypes;
  try {
    RefineInputTypes(graph, inputTypes, refinedTypes);
    ToTensorSSA(graph);
    dumpGraphToFile(graph, "after_tssa.rb");
    HoistLoopInvariants(graph);
    EliminateCommonSubexprTSSA(graph);
    dumpGraphToFile(graph, "after_cse.rb");
    ParallelizeLoops(graph);
    InferDtypeAndDevice(graph, refinedTypes);
    InferShape(graph, refinedTypes);
    dumpGraphToFile(graph, "after_par.rb");
    FuseOps(graph, refinedTypes);
    dumpGraphToFile(graph, "after_fuse.rb");
    SplitParallelMaps(graph, refinedTypes);
    dumpGraphToFile(graph, "after_split.rb");
    MapFunctorToParallization(graph, refinedTypes);
    FusedOpToParallization(graph, refinedTypes);
    dumpGraphToFile(graph, "after_codegen.rb");
    Validate(graph);
  } catch (c10::Error &err) {
    std::cout << err.what();
  }
  // Runtime
  // at::List<at::Tensor> a_list = {
  //     at::ones({1, 85, 1, 1}).to(at::kFloat).cuda() * 0,
  //     at::ones({1, 85, 20, 20}).to(at::kFloat).cuda() * 1,
  //     at::ones({1, 85, 40, 40}).to(at::kFloat).cuda() * 2};
  // at::List<double> b_list = {2.0, 3, 4};
  at::Tensor a = at::ones({1, 85, 1, 1}).to(at::kFloat).cuda() * 2;
  Code code(graph, "");
  Stack input = {"", a};
  torch::jit::InterpreterState(code).run(input);
  auto output_tss_parallel = input[0].toTensor();

  GraphFunction origin_function("simple_simple_loop", origin_graph, nullptr);
  input = {"", a};
  origin_function.run(input);
  auto output_origin = input[0].toTensor();
  std::cout << output_tss_parallel << std::endl;
  std::cout << "Checking Pass: "
            << at::allclose(output_tss_parallel, output_origin) << std::endl;
}
