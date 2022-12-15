#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torchvision/vision.h>

#include "freeze_module.h"
#include "parallelize_loops.h"

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
    ParallelizeLoops(graph);
    // auto block = graph->block();
    // graph->setInsertPoint(block->return_node());
    // auto node = graph->create(sym);
    // auto mapBlock = node->addBlock();
    // mapBlock->addInput()->setType(c10::IntType::get());
    // graph->insertNode(node);
    // node->output(0)->setType(
    //     c10::ListType::create(c10::TensorType::createContiguous(
    //         c10::ScalarType::Float, c10::Device(c10::DeviceType::CPU),
    //         {-1, 4})));

    // auto ty = node->output(0)->type();
    // c10::Device dev(c10::DeviceType::CUDA);
    // auto predMaps = graph->inputs()[1];
    // predMaps->setType(TupleType::create(
    //     {TensorType::createContiguous(c10::ScalarType::Float, dev,
    //                                   {1, 255, 10, 10}),
    //      TensorType::createContiguous(c10::ScalarType::Float, dev,
    //                                   {1, 255, 20, 20}),
    //      TensorType::createContiguous(c10::ScalarType::Float, dev,
    //                                   {1, 255, 40, 40})}));
    // try {
    //     rewriteNode(block, [&](Node *node) -> Node * {
    //         if (node->kind() == aten::__getitem__ &&
    //             node->input(0) == predMaps) {
    //             return graph->create(prim::TupleIndex, node->inputs());
    //         }
    //         if (node->kind() == aten::len && node->input(0) == predMaps) {
    //             auto len = graph->create(prim::Constant);
    //             len->i_(Symbol::attr("value"),
    //             predMaps->type()->cast<TupleType>()->elements().size());
    //             len->output(0)->setType(IntType::get());
    //             return len;
    //         }
    //         return nullptr;
    //     });
    // } catch (c10::Error &e) {
    //     std::cerr << e.what();
    //     return 1;
    // }
    // try {
    //     ConstantPropagation(graph);
    //     PropagateInputShapes(graph);
    // } catch (c10::Error &e) {
    //     std::cerr << e.what();
    //     return 1;
    // } catch (ErrorReport &e) {
    //     std::cerr << e.what();
    //     return 1;
    // }
    // graph->dump();
}
