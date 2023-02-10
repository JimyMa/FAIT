#include "parallelize_loops.h"

#include "type_utils.h"
#include "util/ir.h"
#include "util/traits.h"

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

static Node *splitAt(Node *prevParMap, Node *splitNode, Graph *graph,
                     ValueTypeMap &refinedTypes) {
    // Find straight returns and dependent values of previous map
    std::unordered_set<Value *> prevStraightRets;
    std::vector<Value *> nextDepPrevs;
    auto prevBlock = prevParMap->blocks().front();
    for (auto node = prevBlock->nodes().front(); node != splitNode;
         node = node->next()) {
        for (auto output : node->outputs()) {
            for (auto &use : output->uses()) {
                auto user = use.user;
                if (user->kind() == prim::Return)
                    prevStraightRets.insert(output);
                else if (user == splitNode || user->isAfter(splitNode)) {
                    if (std::find(nextDepPrevs.begin(), nextDepPrevs.end(),
                                  output) == nextDepPrevs.end())
                        nextDepPrevs.push_back(output);
                }
            }
        }
    }

    // Create new parallel map node
    auto numMapOut = prevParMap->outputs().size();
    auto nextParMap = graph->create(prim::ParallelMap, {}, 0);
    nextParMap->insertAfter(prevParMap);
    auto nextBlock = nextParMap->addBlock();

    // Add node inputs and block parameters of the first map to the next map
    std::unordered_map<Value *, Value *> valueMap;
    for (auto i = 0u; i < prevParMap->inputs().size(); i++) {
        auto prevIn = prevParMap->input(i), prevParam = prevBlock->inputs()[i];
        nextParMap->addInput(prevIn);
        auto nextParam = nextBlock->addInput()->setType(prevParam->type());
        transferRefinedType(prevParam, nextParam, refinedTypes);
        valueMap.insert({prevParam, nextParam});
    }

    // Possibly move node outputs and block returns of previous map
    std::vector<Value *> nextRets;
    for (auto i = 0u; i < prevBlock->outputs().size();) {
        auto prevRet = prevBlock->outputs()[i], prevOut = prevParMap->output(i);
        if (!prevStraightRets.count(prevRet)) {
            // move to next map
            nextRets.push_back(prevRet);
            auto nextOut = nextParMap->addOutput()->setType(prevOut->type());
            transferRefinedType(prevOut, nextOut, refinedTypes);
            prevOut->replaceAllUsesWith(nextOut);
            prevBlock->eraseOutput(i);
            prevParMap->eraseOutput(i);
        } else {
            // add to input of next map if it is used by it
            if (std::find(nextDepPrevs.begin(), nextDepPrevs.end(), prevRet) !=
                nextDepPrevs.end()) {
                nextParMap->addInput(prevOut);
                auto nextParam =
                    nextBlock->addInput()->setType(prevRet->type());
                transferRefinedType(prevRet, nextParam, refinedTypes);
                valueMap.insert({prevRet, nextParam});
            }
            i++;  // keep return and output
        }
    }

    // Add dependencies between previous and next maps
    for (auto dep : nextDepPrevs) {
        if (prevStraightRets.count(dep)) continue;
        prevBlock->insertOutput(prevBlock->outputs().size(), dep);
        auto prevOut =
            prevParMap->addOutput()->setType(ListType::create(dep->type()));
        auto refinedListTy =
            createRefinedListType(dep->type(), getParMapTripCount(prevParMap));
        setRefinedType(prevOut, refinedListTy, refinedTypes);
        nextParMap->addInput(prevOut);
        auto nextParam = nextBlock->addInput()->setType(dep->type());
        transferRefinedType(dep, nextParam, refinedTypes);
        valueMap.insert({dep, nextParam});
    }

    // Move nodes beginning from the split point to the new map
    moveNodesToBlock(splitNode, prevBlock->return_node(), nextBlock, graph,
                     valueMap);
    removeDeadRefinedTypes(refinedTypes, graph);

    // Add return values to next block
    for (auto ret : nextRets)
        nextBlock->insertOutput(nextBlock->outputs().size(), valueMap.at(ret));

    return nextParMap;
}

static void splitParallelMap(Node *curParMap, Graph *graph,
                             ValueTypeMap &refinedTypes) {
    // Find fusion group and split parallel map
    auto mapBlock = curParMap->blocks().front();
    for (auto node = mapBlock->nodes().front(); node != mapBlock->return_node();
         node = node->next()) {
        // Check if the node is a fusion group
        if (node->kind() != prim::FusionGroup) continue;

        // Split before the group
        if (node->prev()->kind() != prim::Param) {
            curParMap = splitAt(curParMap, node, graph, refinedTypes);
            mapBlock = curParMap->blocks().front();
            node = mapBlock->param_node()->next();
        }

        // Split after the group
        if (node->next()->kind() != prim::Return) {
            curParMap = splitAt(curParMap, node->next(), graph, refinedTypes);
            mapBlock = curParMap->blocks().front();
            node = mapBlock->param_node();
        }
    }
}

void SplitParallelMaps(const std::shared_ptr<Graph> &graph,
                       ValueTypeMap &refinedTypes) {
    // Find all parallel maps
    std::vector<Node *> parMaps;
    traversePostOrder(graph->block(), [&](Node *node) {
        if (node->kind() == prim::ParallelMap) parMaps.push_back(node);
        return true;
    });

    // Split parallel maps for fusion groups
    for (auto parMap : parMaps)
        splitParallelMap(parMap, graph.get(), refinedTypes);

    // Remove unused map inputs and block parameters
    traversePostOrder(graph->block(), [](Node *node) {
        if (node->kind() != prim::ParallelMap) return true;
        auto block = node->blocks().front();
        for (auto i = 1; i < node->inputs().size();) {
            auto mapIn = node->input(i), blockParam = block->inputs()[i];
            if (!blockParam->hasUses()) {
                node->removeInput(i);
                block->eraseInput(i);
            } else
                i++;
        }
        return true;
    });
    removeDeadRefinedTypes(refinedTypes, graph.get());
}

}  // namespace jit
}  // namespace torch
