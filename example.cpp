#include <torch/csrc/jit/serialization/import.h>
#include <torchvision/vision.h>

#include "passes/freeze_module.h"
#include "passes/fuse_ops.h"
#include "passes/parallelize_loops.h"
#include "passes/tensor_ssa.h"

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
    Freeze(&mod);
    auto graph = mod.get_method("forward").graph();
    ToTensorSSA(graph);
    graph->print(std::ofstream("after_tssa.rb"));
    FuseOps(graph);
    graph->print(std::ofstream("after_fuse.rb"));
}
