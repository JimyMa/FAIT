#include <torch/csrc/jit/serialization/import.h>
#include <torchvision/vision.h>

#include "passes/common_passes.h"
#include "passes/freeze_module.h"
#include "passes/fuse_ops.h"
#include "passes/parallelize_loops.h"
#include "passes/refine_types.h"
#include "passes/tensor_ssa.h"
#include "passes/validate_graph.h"

using namespace torch::jit;

static void dumpGraphToFile(const std::shared_ptr<Graph> &graph,
                            const std::string &path) {
  std::ofstream ofs(path);
  graph->print(ofs);
}

int main(int argc, const char *argv[]) {
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
  std::vector<TypePtr> inputTypes{TupleType::create({
      TensorType::createContiguous(c10::kFloat, c10::kCUDA, {1, 255, 10, 10}),
      TensorType::createContiguous(c10::kFloat, c10::kCUDA, {1, 255, 20, 20}),
      TensorType::createContiguous(c10::kFloat, c10::kCUDA, {1, 255, 40, 40}),
  })};
  ValueTypeMap refinedTypes;
  try {
    Freeze(&mod);
    auto graph = mod.get_method("forward").graph();
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
    ToMutableTensors(graph);
    ConvertInfusibleMapsToLoops(graph, refinedTypes);
    CanonicalizeFusableMaps(graph);
    dumpGraphToFile(graph, "after_back.rb");
    Validate(graph);
  } catch (c10::Error &err) {
    std::cout << err.what();
  }
}
