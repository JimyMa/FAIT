#include "refine_types.h"

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/ir_views.h>

#include "parallelize_loops.h"
#include "tensor_ssa.h"
#include "util/ir.h"

namespace torch {
namespace jit {

static void setRefinedType(Value *value, TypePtr newType,
                           ValueTypeMap &refinedTypes) {
    // Directly set if the new type kind matches the original value type
    auto prevType = value->type();
    auto prevKind = prevType->kind(), newKind = newType->kind();
    if (prevKind == newKind) {
        value->setType(newType);
        return;
    }

    // Set refined type in the map
    refinedTypes.insert({value, newType});

    // Convert refined type to match original type kind
    if (prevKind == TypeKind::ListType && newKind == TypeKind::TupleType) {
        // Unify types in the tuple
        auto elemTypes = newType->cast<TupleType>()->elements();
        auto unified = c10::unifyTypeList(elemTypes, std::cout);
        if (!unified.has_value()) {
            throw c10::TypeError("Cannot unify elements in " + newType->str(),
                                 c10::get_backtrace());
        }
        value->setType(ListType::create(*unified));
    } else {
        throw c10::TypeError("Cannot convert " + newType->str() +
                                 " to refine " + prevType->str(),
                             c10::get_backtrace());
    }
}

void RefineInputTypes(const std::shared_ptr<Graph> &graph,
                      const std::vector<TypePtr> &inputTypes,
                      ValueTypeMap &refinedTypes) {
    // Check if the number of type list matches the graph
    if (graph->inputs().size() != inputTypes.size() + 1) {
        throw c10::TypeError(
            "Expect " + std::to_string(graph->inputs().size() - 1) +
                " types, got " + std::to_string(inputTypes.size()),
            c10::get_backtrace());
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

static void propagateGetItem(TYPE_PROP_PARAMS) {
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

static void propagateListConstruct(TYPE_PROP_PARAMS) {
    std::vector<TypePtr> elemTypes;
    for (auto input : node->inputs()) elemTypes.push_back(input->type());
    setRefinedType(node->output(0), TupleType::create(elemTypes), refinedTypes);
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
            throw c10::TypeError("Cannot unify types " + thenTy->str() +
                                     " and " + elseTy->str() +
                                     " of `If` output",
                                 c10::get_backtrace());
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
        auto inListTy = node->input(i)->type()->cast<ListType>();
        block->inputs()[i]->setType(inListTy->getElementType());
    }

    // Propagate types inside the block
    propFunc(block, refinedTypes);

    // Propagate return types to output lists
    for (auto i = 0u; i < node->outputs().size(); i++) {
        auto ret = block->outputs()[i], outList = node->output(i);
        outList->setType(ListType::create(ret->type()));
    }
}

static void propagateTssaOps(TYPE_PROP_PARAMS) {
    node->output(0)->setType(node->input(0)->type());
}

std::unordered_map<Symbol, void (*)(TYPE_PROP_PARAMS)> symbolPropagators{
    {prim::Constant, propagateConstant},
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
                break;
            }

            default: {
                if (symbolPropagators.count(kind))
                    symbolPropagators[kind](node, refinedTypes, inferDtypeIn);
            }
        }
    }
}

static void inferDeviceIn(Block *block, ValueTypeMap &refinedTypes) {
    auto graph = block->owningGraph();
    for (auto node = block->nodes().front(); node != block->nodes().back();
         node = node->next()) {
        // Handle special symbols
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
                break;
            }

            default: {
                if (symbolPropagators.count(kind))
                    symbolPropagators[kind](node, refinedTypes, inferDeviceIn);
            }
        }
    }
}

void InferDtypeAndDevice(const std::shared_ptr<Graph> &graph,
                         ValueTypeMap &refinedTypes) {
    inferDtypeIn(graph->block(), refinedTypes);
    inferDeviceIn(graph->block(), refinedTypes);
}

}  // namespace jit
}  // namespace torch