#include "refine_types.h"

#include <torch/csrc/jit/ir/ir_views.h>

#include "common_passes.h"
#include "parallelize_loops.h"
#include "tensor_ssa.h"
#include "type_utils.h"
#include "util/ir.h"

namespace torch {
namespace jit {

static TypePtr convertToMatch(TypePtr src, TypePtr tgt) {
    auto srcKind = src->kind(), tgtKind = tgt->kind();
    if (srcKind == TypeKind::ListType && tgtKind == TypeKind::ListType) {
        // Convert element types
        return ListType::create(
            convertToMatch(src->cast<ListType>()->getElementType(),
                           tgt->cast<ListType>()->getElementType()));
    } else if (srcKind == TypeKind::TupleType &&
               tgtKind == TypeKind::ListType) {
        // Unify types in the tuple
        auto elemTypes = src->cast<TupleType>()->elements();
        auto unified = c10::unifyTypeList(elemTypes, std::cout);
        if (!unified.has_value())
            throw typeError("Cannot unify elements in ", *src);
        auto matched =
            convertToMatch(*unified, tgt->cast<ListType>()->getElementType());
        return ListType::create(matched);
    } else if (tgtKind == TypeKind::OptionalType) {
        if (srcKind == TypeKind::OptionalType) {
            return OptionalType::create(
                convertToMatch(src->cast<OptionalType>()->getElementType(),
                               tgt->cast<OptionalType>()->getElementType()));
        } else {
            return OptionalType::create(convertToMatch(
                src, tgt->cast<OptionalType>()->getElementType()));
        }
    } else if (srcKind == tgtKind)
        return src;
    else
        throw typeError("Cannot convert ", *src, " to match ", *tgt);
}

void setRefinedType(Value *value, const TypePtr &newType,
                    ValueTypeMap &refinedTypes) {
    auto &uses = value->uses();
    TypePtr matched = nullptr;
    if (value->type()->kind() == TypeKind::ListType &&
        std::any_of(uses.begin(), uses.end(),
                    [](const Use &use) { return use.user->maybeSchema(); })) {
        matched = value->type();
    } else {
        matched = convertToMatch(newType, value->type());
        value->setType(matched);
    }
    if (*matched != *newType) refinedTypes[value] = newType;
}

static void markLiveValues(Block *block,
                           std::unordered_set<Value *> &deadValues) {
    for (auto param : block->inputs()) deadValues.erase(param);
    for (auto node : block->nodes()) {
        for (auto subBlock : node->blocks())
            markLiveValues(subBlock, deadValues);
        for (auto output : node->outputs()) deadValues.erase(output);
    }
}

void removeDeadRefinedTypes(ValueTypeMap &refinedTypes, Graph *graph) {
    std::unordered_set<Value *> deadValues;
    for (auto &pair : refinedTypes) deadValues.insert(pair.first);
    markLiveValues(graph->block(), deadValues);
    for (auto dead : deadValues) refinedTypes.erase(dead);
}

void RefineInputTypes(const std::shared_ptr<Graph> &graph,
                      const std::vector<TypePtr> &inputTypes,
                      ValueTypeMap &refinedTypes) {
    // Check if the number of type list matches the graph
    if (graph->inputs().size() != inputTypes.size() + 1) {
        throw typeError("Expect ", graph->inputs().size() - 1, " types, got ",
                        inputTypes.size());
    }

    // Refine input types
    for (auto i = 0u; i < inputTypes.size(); i++)
        setRefinedType(graph->inputs()[i + 1], inputTypes[i], refinedTypes);
}

using BlockPropagator = std::function<void(Block *, ValueTypeMap &)>;

#define TYPE_PROP_PARAMS \
    Node *node, ValueTypeMap &refinedTypes, const BlockPropagator &propFunc

static void propagateConstant(TYPE_PROP_PARAMS) {
    auto ty = node->output(0)->type();
    if (ty->kind() == TypeKind::ListType) {
        auto listVal = node->ival(attr::value).toListRef();
        std::vector<TypePtr> elemTypes;
        for (auto elem : listVal) elemTypes.push_back(elem.type());
        ty = TupleType::create(elemTypes);
    }
    setRefinedType(node->output(0), ty, refinedTypes);
}

static void propagateTupleConstruct(TYPE_PROP_PARAMS) {
    std::vector<TypePtr> elemTypes;
    for (auto input : node->inputs()) elemTypes.push_back(input->type());
    node->output(0)->setType(TupleType::create(elemTypes));
}

static void propagateTupleUnpack(TYPE_PROP_PARAMS) {
    auto elemTypes = node->input(0)->type()->cast<TupleType>()->elements();
    for (auto i = 0u; i < elemTypes.size(); i++)
        node->output(i)->setType(elemTypes[i]);
}

static void propagateListUnpack(TYPE_PROP_PARAMS) {
    auto list = node->input(0);
    if (refinedTypes.count(list) &&
        refinedTypes[list]->kind() == TypeKind::TupleType) {
        auto elemTypes = refinedTypes[list]->cast<TupleType>()->elements();
        for (auto i = 0u; i < elemTypes.size(); i++)
            node->output(i)->setType(elemTypes[i]);
    } else {
        auto elemTy = list->type()->cast<ListType>()->getElementType();
        for (auto output : node->outputs()) output->setType(elemTy);
    }
}

static void propagateListConstruct(TYPE_PROP_PARAMS) {
    std::vector<TypePtr> elemTypes;
    for (auto input : node->inputs()) elemTypes.push_back(input->type());
    setRefinedType(node->output(0), TupleType::create(elemTypes), refinedTypes);
}

static void propagateGetItem(TYPE_PROP_PARAMS) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(node->maybeSchema());
    auto list = node->input(0), index = node->input(1);
    if (refinedTypes.count(list) && refinedTypes[list]->castRaw<TupleType>() &&
        index->node()->kind() == prim::Constant) {
        auto elems = refinedTypes[list]->cast<TupleType>()->elements();
        auto idxCnst = index->node()->i(attr::value);
        node->output(0)->setType(elems.at(idxCnst));
    } else {
        auto elemTy = list->type()->cast<ListType>()->getElementType();
        node->output(0)->setType(elemTy);
    }
}

static void propagateIf(TYPE_PROP_PARAMS) {
    IfView view(node);
    propFunc(view.thenBlock(), refinedTypes);
    propFunc(view.elseBlock(), refinedTypes);
    for (auto i = 0u; i < view.outputs().size(); i++) {
        auto thenTy = view.thenOutputs()[i]->type(),
             elseTy = view.elseOutputs()[i]->type();
        auto outType = c10::unifyTypes(thenTy, elseTy);
        if (!outType) {
            throw typeError("Cannot unify types ", *thenTy, " and ", *elseTy,
                            " of the ", i, "-th `If` output");
        }
        view.outputs()[i]->setType(*outType);
    }
}

static void propagateLoop(TYPE_PROP_PARAMS) {
    // Propagate inputs to loop parameters
    LoopView view(node);
    auto numCarried = view.carriedInputs().size();
    for (auto i = 0u; i < numCarried; i++) {
        auto input = view.carriedInputs()[i],
             param = view.bodyCarriedInputs()[i];
        param->setType(input->type());
    }

    // Propagate carried values until a fixed point
    while (true) {
        propFunc(view.bodyBlock(), refinedTypes);
        bool changed = false;
        for (auto i = 0u; i < numCarried; i++) {
            auto param = view.bodyCarriedInputs()[i],
                 ret = view.bodyCarriedOutputs()[i];
            if (*param->type() != *ret->type()) {
                param->setType(ret->type());
                changed = true;
            }
        }
        if (!changed) break;
    }

    // Propagate returns to outputs
    for (auto i = 0u; i < numCarried; i++) {
        auto ret = view.bodyCarriedOutputs()[i],
             output = view.carriedOutputs()[i];
        output->setType(ret->type());
    }
}

static void propagateParallelMap(TYPE_PROP_PARAMS) {
    // Propagate element types of input lists to block parameters
    auto block = node->blocks().front();
    for (auto i = 1u; i < node->inputs().size(); i++) {
        auto inListTy = getRefinedType(node->input(i), refinedTypes);
        block->inputs()[i]->setType(getUnifiedElementType(inListTy));
    }

    // Propagate types inside the block
    propFunc(block, refinedTypes);

    // Propagate return types to output lists
    auto lenIVal = toIValue(node->input(0));
    auto len = mapOpt<size_t>(
        lenIVal, [](const IValue &ival) { return size_t(ival.toInt()); });
    for (auto i = 0u; i < node->outputs().size(); i++) {
        auto ret = block->outputs()[i], outList = node->output(i);
        setRefinedType(outList, createRefinedListType(ret->type(), len),
                       refinedTypes);
    }
}

static void propagateTssaOps(TYPE_PROP_PARAMS) {
    node->output(0)->setType(node->input(0)->type());
}

std::unordered_map<Symbol, void (*)(TYPE_PROP_PARAMS)> symbolPropagators{
    {prim::Constant, propagateConstant},
    {prim::TupleConstruct, propagateTupleConstruct},
    {prim::TupleUnpack, propagateTupleUnpack},
    {prim::ListConstruct, propagateListConstruct},
    {prim::ListUnpack, propagateListUnpack},
    {aten::__getitem__, propagateGetItem},
    {prim::If, propagateIf},
    {prim::Loop, propagateLoop},
    {prim::ParallelMap, propagateParallelMap},
    {tssa::Assign, propagateTssaOps},
    {tssa::Update, propagateTssaOps},
};

static void inferDtypeIn(Block *block, ValueTypeMap &refinedTypes) {
    auto graph = block->owningGraph();
    for (auto node = block->nodes().front(); node != block->nodes().back();
         node = node->next()) {
        // Handle special symbols
        auto kind = node->kind();
        switch (node->kind()) {
            case prim::dtype: {
                auto dtype =
                    node->input(0)->type()->cast<TensorType>()->scalarType();
                if (dtype) {
                    graph->setInsertPoint(node->next());
                    auto cnstVal =
                        graph->insertConstant(int64_t(dtype.value()));
                    node->output(0)->replaceAllUsesWith(cnstVal);
                    node = remove(node);
                }
            } break;

            default: {
                if (symbolPropagators.count(kind)) {
                    symbolPropagators[kind](node, refinedTypes, inferDtypeIn);
                    continue;
                }

                // Skip if there is no tensor in the output
                auto outputs = node->outputs();
                if (std::none_of(outputs.begin(), outputs.end(), isTensor))
                    continue;

                // Use per-operator dtype function to infer dtype
                auto dtype = c10::kFloat;
                auto op = node->maybeOperator();
                if (op && dtypeFuncs.contains(*op))
                    dtype = (*dtypeFuncs.find(*op))(node, refinedTypes);
                else {
                    for (auto input : node->inputs()) {
                        if (!isTensor(input)) continue;
                        auto inDtype =
                            input->type()->cast<TensorType>()->scalarType();
                        if (inDtype) {
                            dtype = *inDtype;
                            break;
                        }
                    }
                }

                // Propagate device to outputs
                for (auto output : outputs) {
                    if (!isTensor(output)) continue;
                    output->setType(
                        output->type()->cast<TensorType>()->withScalarType(
                            dtype));
                }
            } break;
        }
    }
}

static void inferDeviceIn(Block *block, ValueTypeMap &refinedTypes) {
    auto graph = block->owningGraph();
    for (auto node = block->nodes().front(); node != block->nodes().back();
         node = node->next()) {
        auto kind = node->kind();
        switch (node->kind()) {
            case prim::device: {
                auto device =
                    node->input(0)->type()->cast<TensorType>()->device();
                if (device) {
                    graph->setInsertPoint(node->next());
                    auto cnstVal = graph->insertConstant(device.value());
                    node->output(0)->replaceAllUsesWith(cnstVal);
                    node = remove(node);
                }
            } break;

            default: {
                // Propagate types for special symbols
                if (symbolPropagators.count(kind)) {
                    symbolPropagators[kind](node, refinedTypes, inferDeviceIn);
                    continue;
                }

                // Skip if there is no tensor in the output
                auto outputs = node->outputs();
                if (std::none_of(outputs.begin(), outputs.end(), isTensor))
                    continue;

                // Use per-operator device function to infer device
                c10::Device device(c10::kCUDA);
                auto op = node->maybeOperator();
                if (op && deviceFuncs.contains(*op))
                    device = (*deviceFuncs.find(*op))(node, refinedTypes);
                else {
                    for (auto input : node->inputs()) {
                        if (!isTensor(input)) continue;
                        auto inputDev =
                            input->type()->cast<TensorType>()->device();
                        if (inputDev) {
                            device = *inputDev;
                            break;
                        }
                    }
                }

                // Propagate device to outputs
                for (auto output : outputs) {
                    if (!isTensor(output)) continue;
                    output->setType(
                        output->type()->cast<TensorType>()->withDevice(device));
                }
            }
        }
    }
}

void InferDtypeAndDevice(const std::shared_ptr<Graph> &graph,
                         ValueTypeMap &refinedTypes) {
    initTensorTypeFuncs();
    inferDeviceIn(graph->block(), refinedTypes);
    inferDtypeIn(graph->block(), refinedTypes);
}

static void inferShapeIn(Block *block, ValueTypeMap &refinedTypes) {
    auto graph = block->owningGraph();
    for (auto node = block->nodes().front(); node != block->nodes().back();
         node = node->next()) {
        // Handle special symbols
        auto kind = node->kind();
        switch (node->kind()) {
            case aten::size: {
                auto tensorTy = node->input(0)->type()->cast<TensorType>();
                auto shape = tensorTy->sizes();
                graph->setInsertPoint(node->next());
                Value *cnstVal = nullptr;
                if (node->inputs().size() == 1 && shape.isComplete()) {
                    cnstVal = graph->insertConstant(*shape.concrete_sizes());
                } else if (node->inputs().size() == 2 && shape.size()) {
                    auto index = toIValue(node->input(1));
                    if (!index || !shape[index->toInt()]) continue;
                    cnstVal = graph->insertConstant(shape[index->toInt()]);
                }
                if (cnstVal) {
                    node->output(0)->replaceAllUsesWith(cnstVal);
                    node = remove(node);
                }
            } break;

            case aten::len: {
                auto input = node->input(0);
                auto inTy = input->type();
                c10::optional<int64_t> len;
                if (inTy->kind() == TypeKind::TensorType) {
                    auto sizes = inTy->cast<TensorType>()->sizes();
                    if (sizes.size()) len = sizes[0];
                } else if (inTy->kind() == TypeKind::ListType &&
                           refinedTypes.count(input) &&
                           refinedTypes[input]->kind() == TypeKind::TupleType) {
                    len = refinedTypes[input]
                              ->cast<TupleType>()
                              ->elements()
                              .size();
                }
                if (len) {
                    graph->setInsertPoint(node->next());
                    auto cnstVal = graph->insertConstant(*len);
                    node->output(0)->replaceAllUsesWith(cnstVal);
                    node = remove(node);
                }
            } break;

            default: {
                if (symbolPropagators.count(kind)) {
                    symbolPropagators[kind](node, refinedTypes, inferShapeIn);
                    continue;
                }

                // Skip if there is no tensor in the output
                auto outputs = node->outputs();
                if (std::none_of(outputs.begin(), outputs.end(), isTensor))
                    continue;

                // Use per-operator shape function to infer shape
                auto op = node->maybeOperator();
                if (!(op && shapeFuncs.contains(*op))) continue;
                auto shape = (*shapeFuncs.find(*op))(node, refinedTypes);
                for (auto output : outputs) {
                    if (!isTensor(output)) continue;
                    output->setType(
                        output->type()->cast<TensorType>()->withSymbolicShapes(
                            shape));
                }
            } break;
        }
    }
}

void InferShape(const std::shared_ptr<Graph> &graph,
                ValueTypeMap &refinedTypes) {
    initTensorTypeFuncs();
    while (true) {
        inferShapeIn(graph->block(), refinedTypes);
        if (!FoldConstantsTSSA(graph)) break;
        EliminateDeadCodeTSSA(graph);
    }
    HoistLoopInvariants(graph);
    EliminateCommonSubexprTSSA(graph);
}

}  // namespace jit
}  // namespace torch