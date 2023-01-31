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
    auto mapArgs = std::move(inLists);
    mapArgs.insert(mapArgs.begin(), loop->input(0));
    auto mapNode = graph->create(prim::ParallelMap, mapArgs, outLists.size());
    mapNode->setSourceRange(loop->sourceRange());
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
}

static Node *splitAt(Node *prevParMap, Node *firstParMap, Node *splitNode,
                     const std::vector<Value *> &depValues, Graph *graph) {
    // Create new parallel map node
    auto numMapOut = prevParMap->outputs().size();
    auto nextParMap = graph->create(prim::ParallelMap, {}, numMapOut);
    nextParMap->insertAfter(prevParMap);

    // Transfer outputs of the map
    for (auto i = 0u; i < numMapOut; i++) {
        auto prevMapOut = prevParMap->output(i),
             nextMapOut = nextParMap->output(i);
        nextMapOut->setType(prevMapOut->type());
        prevMapOut->replaceAllUsesWith(nextMapOut);
    }

    // Reset outputs of previous map
    prevParMap->removeAllOutputs();
    std::vector<Value *> prevMapOutputs;
    for (auto dep : depValues) {
        auto prevMapOut = prevParMap->addOutput();
        prevMapOut->setType(ListType::create(dep->type()));
        prevMapOutputs.push_back(prevMapOut);
    }

    // Reset block output of the previous map
    auto prevBlock = prevParMap->blocks().front();
    auto origBlockRets = prevBlock->outputs().vec();
    prevBlock->removeAllOutputs();
    for (auto dep : depValues)
        prevBlock->insertOutput(prevBlock->outputs().size(), dep);

    // Add inputs to the next map
    for (auto firstMapIn : firstParMap->inputs())
        nextParMap->addInput(firstMapIn);
    for (auto prevMapOut : prevMapOutputs) nextParMap->addInput(prevMapOut);

    // Add block parameters to the next map
    std::unordered_map<Value *, Value *> valueMap;
    auto nextBlock = nextParMap->addBlock();
    for (auto firstBlockParam : firstParMap->inputs()) {
        auto nextBlockParam = nextBlock->addInput();
        nextBlockParam->setType(firstBlockParam->type());
        valueMap.insert({firstBlockParam, nextBlockParam});
    }
    for (auto dep : depValues) {
        auto nextBlockParam = nextBlock->addInput();
        nextBlockParam->setType(dep->type());
        valueMap.insert({dep, nextBlockParam});
    }

    // Move nodes beginning from the split point to the new map
    moveNodesToBlock(splitNode, prevBlock->return_node(), nextBlock, graph,
                     valueMap);

    // Add block return to the new map
    for (auto ret : origBlockRets)
        nextBlock->insertOutput(nextBlock->outputs().size(), valueMap.at(ret));

    return nextParMap;
}

static void splitParallelMap(Node *firstParMap, Graph *graph) {
    // Find fusion group and split parallel map
    auto curParMap = firstParMap;
    auto mapBlock = curParMap->blocks().front();
    for (auto node = mapBlock->nodes().front(); node != mapBlock->return_node();
         node = node->next()) {
        // Check if the node is a fusion group
        if (node->kind() != prim::FusionGroup) continue;

        // Split before the group
        if (node->prev()->kind() != prim::Param) {
            // Collect values this group depends in addition to parameters of
            // the map
            std::vector<Value *> groupDeps;
            for (auto input : node->inputs()) {
                if (input->node()->kind() == prim::Param) continue;
                if (input->node()->owningBlock() != mapBlock) continue;
                groupDeps.push_back(input);
            }

            // Split the parallel map
            auto nextParMap =
                splitAt(curParMap, firstParMap, node, groupDeps, graph);

            // Update block and node pointers
            curParMap = nextParMap;
            mapBlock = curParMap->blocks().front();
            node = mapBlock->param_node()->next();
        }

        // Split after the group
        if (node->next()->kind() != prim::Return) {
            // Collect all values this group outputs
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(node->kind() == prim::FusionGroup);
            auto groupOutputs = node->outputs().vec();

            // Split the group
            auto nextParMap = splitAt(curParMap, firstParMap, node->next(),
                                      groupOutputs, graph);

            // Update block and node pointers
            curParMap = nextParMap;
            mapBlock = curParMap->blocks().front();
            node = mapBlock->param_node();
        }
    }
}

void SplitParallelMaps(const std::shared_ptr<Graph> &graph) {
    // Find all parallel maps
    std::vector<Node *> parMaps;
    traversePostOrder(graph->block(), [&](Node *node) {
        if (node->kind() == prim::ParallelMap) parMaps.push_back(node);
        return true;
    });

    // Split parallel maps for fusion groups
    for (auto parMap : parMaps) splitParallelMap(parMap, graph.get());
}

}  // namespace jit
}  // namespace torch
