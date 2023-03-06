#include <ATen/Context.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAFunctions.h>

#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torchvision/vision.h>

#include "passes/common_passes.h"
#include "passes/freeze_module.h"
#include "passes/fuse_ops.h"
#include "passes/parallelize_loops.h"
#include "passes/refine_types.h"
#include "passes/tensor_ssa.h"
#include "passes/validate_graph.h"
#include "passes/te_op.h"


using namespace torch::jit;

static void dumpGraphToFile(const std::shared_ptr<Graph> &graph,
                            const std::string &path) {
    std::ofstream ofs(path);
    graph->print(ofs);
}

int main(int argc, const char *argv[]) {
    at::globalContext().lazyInitCUDA();
    std::cout << "current device: " << int(at::cuda::current_device()) << std::endl;
    std::cout << "device count: " << int(at::cuda::device_count()) << std::endl;
    
    


    // Runtime
    at::List<at::Tensor> a_list = {at::ones({1, 85, 1, 1}).to(at::kFloat).cuda() * 0,
                               at::ones({1, 85, 20, 20}).to(at::kFloat).cuda() * 1,
                               at::ones({1, 85, 40, 40}).to(at::kFloat).cuda() * 2};
    // auto num_gpus = c10::cuda::device_count();
    // std::cout << num_gpus << std::endl;

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
    std::vector<TypePtr> inputTypes{TupleType::create({
        TensorType::createContiguous(c10::kFloat, c10::kCUDA, {1, 85, 1, 1}),
        TensorType::createContiguous(c10::kFloat, c10::kCUDA, {1, 85, 20, 20}),
        TensorType::createContiguous(c10::kFloat, c10::kCUDA, {1, 85, 40, 40}),
    })};
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
        dumpGraphToFile(graph, "after_codegen.rb");
        Validate(graph);
    } catch (c10::Error &err) {
        std::cout << err.what();
    }
    
    // at::List<double> b_list = {2.0, 3, 4};
    Code code(graph, "");
    Stack input = {"", a_list};
    torch::jit::InterpreterState(code).run(input);
    std::cout << input[0].toTensorList()[0] << std::endl;
    // std::cout << input[0].toTensorList()[1] << std::endl;
    // std::cout << input[0].toTensorList()[2] << std::endl;
    // for (int i = 0; i < 3; i++) {
    //     input = {"", a_list, b_list};
    //     InterpreterState(code).run(input);
    //     std::cout << input[0].toTensorList()[0] << std::endl;
    //     std::cout << input[0].toTensorList()[1] << std::endl;
    // }
    
}
