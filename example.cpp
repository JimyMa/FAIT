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
    std::unordered_map<Value *, TypePtr> refinedTypes;
    try {
        Freeze(&mod);
        auto graph = mod.get_method("forward").graph();
        RefineInputTypes(graph, inputTypes, refinedTypes);
        ToTensorSSA(graph);
        graph->print(std::ofstream("after_tssa.rb"));
        HoistLoopInvariants(graph);
        EliminateCommonSubexprTSSA(graph);
        graph->print(std::ofstream("after_cse.rb"));
        ParallelizeLoops(graph);
        InferDtypeAndDevice(graph, refinedTypes);
        graph->print(std::ofstream("after_par.rb"));
        FuseOps(graph);
        graph->print(std::ofstream("after_fuse.rb"));
        SplitParallelMaps(graph);
        graph->print(std::ofstream("after_split.rb"));
        Validate(graph);
    } catch (c10::Error &err) {
        std::cout << err.what();
    }
}
