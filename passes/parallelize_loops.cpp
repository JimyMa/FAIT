#include "parallelize_loops.h"

#include "util/ir.h"
#include "util/traits.h"

namespace c10 {
namespace prim {

Symbol ParallelMap = Symbol::prim("ParallelMap");

}
}  // namespace c10

namespace torch {
namespace jit {

void markNonParLoops(Block *block, Node *loop,
                     std::unordered_set<Node *> &nonParLoops) {
    for (auto node : block->nodes()) {
        // Reject loop with value dependency
        auto isLoop = node->kind() == prim::Loop;
        if (isLoop && node->inputs().size() > 2) nonParLoops.insert(node);

        // Recursively visit blocks
        for (auto nested : node->blocks())
            markNonParLoops(nested, isLoop ? node : loop, nonParLoops);

        // Check list operation
        if (node->inputs().empty()) continue;
        auto list = node->input(0);
        if (!list->type()->cast<ListType>()) continue;

        // No need to care about local lists
        auto listDefBlock = list->node()->owningBlock();
        if (listDefBlock == block) continue;

        // Reject if the definition of the list is not in the outer block
        auto owningNode = block->owningNode();
        if (owningNode && owningNode->owningBlock() != listDefBlock) {
            // all the loops along the way are not parallelizable
            for (auto b = block; b != listDefBlock;
                 b = b->owningNode()->owningBlock()) {
                auto n = b->owningNode();
                if (n->kind() == prim::Loop) nonParLoops.insert(n);
            }
        }

        // Check if the read (`aten::__getitem__`) is supported
        if (!loop) continue;
        auto body = loop->blocks()[0];
        if (node->kind() == aten::__getitem__) {
            auto iterVar = body->inputs()[0];
            auto index = node->input(1);
            if (index->node()->kind() != prim::Constant && index != iterVar) {
                nonParLoops.insert(loop);
                continue;
            }
        }

        // Check if there is only one `aten::append` to the list
        auto numAppend = 0u;
        for (auto &use : list->uses()) {
            auto user = use.user;
            if (!isMutating(user)) continue;
            if (user->kind() != aten::append) {
                nonParLoops.insert(loop);
                continue;
            }
            numAppend++;
        }
        if (numAppend > 1) nonParLoops.insert(loop);
    }
}

void convertLoopToMap(Node *loop, Graph *graph) {
    // Collect input and output lists of the loop
    std::vector<Value *> inLists, outLists;
    std::vector<Value *> inElems, outElems;
    auto body = loop->blocks()[0];
    auto iterVar = body->inputs()[0];
    for (auto node : body->nodes()) {
        switch (node->kind()) {
            case aten::__getitem__: {
                if (node->input(1) != iterVar) continue;
                inLists.push_back(node->input(0));
                inElems.push_back(node->output(0));
                break;
            }

            case aten::append: {
                outLists.push_back(node->input(0));
                outElems.push_back(node->input(1));
                break;
            }
        }
    }

    // Create map node
    auto mapNode = graph->create(prim::ParallelMap, inLists, outLists.size());
    mapNode->insertAfter(loop);
    auto mapBlock = mapNode->addBlock();
    mapBlock->cloneFrom(body, [](Value *v) { return v; });
    auto mapIdx = mapBlock->inputs()[0];

    // Process nodes in the body and remap values
    for (auto iter = mapBlock->nodes().begin(); iter != mapBlock->nodes().end();
         ++iter) {
        auto node = *iter;
        switch (node->kind()) {
            case aten::__getitem__: {
                if (node->input(1) != mapIdx) continue;
                auto param = mapBlock->addInput();
                param->setType(node->output(0)->type());
                node->output(0)->replaceAllUsesWith(param);
                iter.destroyCurrent();
                break;
            }

            case aten::append: {
                auto ret = node->input(1);
                mapBlock->insertOutput(mapBlock->outputs().size(), ret);
                iter.destroyCurrent();
                break;
            }
        }
    }
    mapBlock->eraseOutput(0);

    // Remap output values of the map node
    for (auto i = 0u; i < outLists.size(); i++) {
        auto mapOut = mapNode->output(i), loopOut = outLists[i];
        mapOut->setType(loopOut->type());
        loopOut->replaceAllUsesAfterNodeWith(loop, mapOut);
    }

    // Remove loop node and output list definitions
    loop->destroy();
    for (auto outList : outLists) outList->node()->destroy();
}

void ParallelizeLoops(const std::shared_ptr<Graph> &graph) {
    // Find parallelizable loops
    std::unordered_set<Node *> nonParLoops;
    markNonParLoops(graph->block(), nullptr, nonParLoops);
    std::vector<Node *> loops;
    traversePostOrder(graph->block(), [&](Node *node) {
        if (node->kind() == prim::Loop && !nonParLoops.count(node))
            loops.push_back(node);
        return true;
    });

    // Convert to parallel maps
    for (auto loop : loops) convertLoopToMap(loop, graph.get());

    return;
}

}  // namespace jit
}  // namespace torch
