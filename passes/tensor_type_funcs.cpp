#include "refine_types.h"

namespace torch {
namespace jit {

#define INFER_PARAMS Node *node, ValueTypeMap &refinedTypes

static OperatorSet creationOps{
    "aten::tensor.float(float t, *, ScalarType? dtype=None, Device? "
    "device=None, bool requires_grad=False) -> Tensor",
    "aten::tensor.int(int t, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
    "aten::tensor.bool(bool t, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
    "aten::tensor(t[] data, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
    "aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool "
    "non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> "
    "Tensor(a)",
    "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, "
    "bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
    "aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, "
    "bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
    "aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, "
    "Device? device=None, bool? pin_memory=None) -> Tensor",
    "aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, "
    "Layout? layout=None, Device? device=None, bool? pin_memory=None) -> "
    "Tensor",
    "aten::arange.start_step(Scalar start, Scalar end, Scalar step=1, *, "
    "ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? "
    "pin_memory=None) -> Tensor",
    "aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? "
    "layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
    "aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? "
    "layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? "
    "memory_format=None) -> Tensor",
    "aten::new_zeros(Tensor self, SymInt[] size, *, ScalarType? dtype=None, "
    "Layout? layout=None, Device? device=None, bool? pin_memory=None) -> "
    "Tensor",
};

static c10::Device inferDeviceCreationOps(INFER_PARAMS) {
    // Check if there is source tensor (`self`)
    c10::Device device(c10::kCUDA);
    auto &schema = node->schema();
    auto selfIdx = schema.argumentIndexWithName("self");
    if (selfIdx)
        device = *node->input(*selfIdx)->type()->cast<TensorType>()->device();

    // Check if there is target tensor (`other`)
    auto otherIdx = schema.argumentIndexWithName("other");
    if (otherIdx)
        device = *node->input(*otherIdx)->type()->cast<TensorType>()->device();

    // Check if device is specified as an argument
    auto deviceIdx = schema.argumentIndexWithName("device");
    if (deviceIdx) {
        auto deviceArg = node->input(*deviceIdx);
        auto ival = toIValue(node->input(*deviceIdx));
        if (ival && ival->isDevice()) device = (*ival).toDevice();
    }

    return device;
}

static OperatorSet convertOrFillOps{
    "aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool "
    "non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> "
    "Tensor(a)",
    "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, "
    "bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
    "aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, "
    "bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
    "aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, "
    "Device? device=None, bool? pin_memory=None) -> Tensor",
    "aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, "
    "Layout? layout=None, Device? device=None, bool? pin_memory=None) -> "
    "Tensor",
    "aten::arange.start_step(Scalar start, Scalar end, Scalar step=1, *, "
    "ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? "
    "pin_memory=None) -> Tensor",
    "aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? "
    "layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
    "aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? "
    "layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? "
    "memory_format=None) -> Tensor",
    "aten::new_zeros(Tensor self, SymInt[] size, *, ScalarType? dtype=None, "
    "Layout? layout=None, Device? device=None, bool? pin_memory=None) -> "
    "Tensor",
};

static void updateDtypeFromArgs(Node *node, const FunctionSchema &schema,
                                c10::ScalarType &dtype) {
    // Check if there is source tensor (`self`)
    auto selfIdx = schema.argumentIndexWithName("self");
    if (selfIdx)
        dtype =
            *node->input(*selfIdx)->type()->cast<TensorType>()->scalarType();

    // Check if there is target tensor (`other`)
    auto otherIdx = schema.argumentIndexWithName("other");
    if (otherIdx)
        dtype =
            *node->input(*otherIdx)->type()->cast<TensorType>()->scalarType();

    // Check if device is specified as an argument
    auto dtypeIdx = schema.argumentIndexWithName("dtype");
    if (dtypeIdx) {
        auto dtypeArg = node->input(*dtypeIdx);
        auto ival = toIValue(node->input(*dtypeIdx));
        if (ival && ival->isInt()) dtype = c10::ScalarType((*ival).toInt());
    }
}

static c10::ScalarType inferDtypeConvertOrFillOps(INFER_PARAMS) {
    auto dtype = c10::kFloat;
    auto &schema = node->schema();
    updateDtypeFromArgs(node, schema, dtype);
    return dtype;
};

static OperatorSet atenTensorOps{
    "aten::tensor.float(float t, *, ScalarType? dtype=None, Device? "
    "device=None, bool requires_grad=False) -> Tensor",
    "aten::tensor.int(int t, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
    "aten::tensor.bool(bool t, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
    "aten::tensor(t[] data, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
};

static c10::optional<size_t> getListLen(Value *list,
                                        ValueTypeMap &refinedTypes) {
    if (refinedTypes.count(list) &&
        refinedTypes[list]->kind() == TypeKind::TupleType) {
        return refinedTypes[list]->cast<TupleType>()->elements().size();
    } else
        return c10::nullopt;
}

static c10::SymbolicShape inferShapeAtenTensorOps(INFER_PARAMS) {
    auto value = node->input(0);
    auto type = value->type();
    if (value->type()->kind() == TypeKind::ListType) {
        auto len = getListLen(value, refinedTypes);
        if (len) {
            return c10::IntArrayRef({int64_t(*len)});
        } else
            return c10::optional<size_t>(1);
    } else {
        return c10::optional<size_t>(0);
    }
}

static std::unordered_map<TypeKind, c10::ScalarType> typeKindsToScalarTypes{
    {TypeKind::FloatType, c10::kFloat},
    {TypeKind::IntType, c10::kLong},
    {TypeKind::BoolType, c10::kBool},
};

static c10::ScalarType inferDtypeAtenTensorOps(INFER_PARAMS) {
    auto value = node->input(0);
    auto type = value->type();
    auto kind = type->kind();
    if (typeKindsToScalarTypes.count(kind))
        return typeKindsToScalarTypes[kind];
    else if (kind == TypeKind::ListType) {
        auto elemTy = type->cast<ListType>()->getElementType();
        TORCH_INTERNAL_ASSERT(typeKindsToScalarTypes.count(elemTy->kind()));
        return typeKindsToScalarTypes[elemTy->kind()];
    } else {
        throw c10::TypeError("Cannot infer data type for input %" +
                                 value->debugName() + " of `aten::tensor`",
                             c10::get_backtrace());
    }
}

static OperatorSet combineOps{
    "aten::cat(Tensor[] tensors, int dim=0) -> Tensor",
    "aten::stack(Tensor[] tensors, int dim=0) -> Tensor",
};

static c10::ScalarType inferDtypeCombineOps(INFER_PARAMS) {
    auto listTy = refinedTypes.at(node->input(0));
    auto tensorTy = listTy->containedType(0)->cast<TensorType>();
    return *tensorTy->scalarType();
}

static c10::Device inferDeviceCombineOps(INFER_PARAMS) {
    auto listTy = refinedTypes.at(node->input(0));
    auto tensorTy = listTy->containedType(0)->cast<TensorType>();
    return *tensorTy->device();
}

static OperatorSet sameShapeOps{
    "aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool "
    "non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> "
    "Tensor(a)",
    "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, "
    "bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
    "aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, "
    "bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
    "aten::exp(Tensor self) -> Tensor",
    "aten::log(Tensor self) -> Tensor",
    "aten::sin(Tensor self) -> Tensor",
    "aten::cos(Tensor self) -> Tensor",
    "aten::sqrt(Tensor self) -> Tensor",
    "aten::sigmoid(Tensor self) -> Tensor",
    "aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor",
    "aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> "
    "Tensor",
    "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::div.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::eq.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::ne.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::gt.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::ge.Scalar(Tensor self, Scalar other) -> Tensor",
};

static c10::SymbolicShape passSameShape(INFER_PARAMS) {
    return node->input(0)->type()->cast<TensorType>()->symbolic_sizes();
}

static OperatorSet rankOneOps{
    "aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, "
    "Device? device=None, bool? pin_memory=None) -> Tensor",
    "aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, "
    "Layout? layout=None, Device? device=None, bool? pin_memory=None) -> "
    "Tensor",
    "aten::arange.start_step(Scalar start, Scalar end, Scalar step=1, *, "
    "ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? "
    "pin_memory=None) -> Tensor",
};

template <size_t Rank>
static c10::SymbolicShape getRankedShape(INFER_PARAMS) {
    return c10::optional<size_t>(Rank);
}

static OperatorSet boolOps{
    "aten::eq.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::eq.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::ne.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::ne.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::lt.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::gt.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::gt.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::ge.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::ge.Scalar(Tensor self, Scalar other) -> Tensor",
};

static OperatorSet longOps{
    "aten::nonzero(Tensor self) -> Tensor",
    "torchvision::nms(Tensor dets, Tensor scores, float iou_threshold) -> "
    "Tensor",
};

static std::initializer_list<
    std::pair<OperatorSet, c10::SymbolicShape (*)(INFER_PARAMS)>>
    shapeFuncInit{
        {atenTensorOps, inferShapeAtenTensorOps},
        {sameShapeOps, passSameShape},
        {rankOneOps, getRankedShape<1>},
    };

static std::initializer_list<
    std::pair<OperatorSet, c10::ScalarType (*)(INFER_PARAMS)>>
    dtypeFuncInit{
        {convertOrFillOps, inferDtypeConvertOrFillOps},
        {atenTensorOps, inferDtypeAtenTensorOps},
        {combineOps, inferDtypeCombineOps},
        {boolOps, [](INFER_PARAMS) { return c10::kBool; }},
        {longOps, [](INFER_PARAMS) { return c10::kLong; }},
    };

static std::initializer_list<
    std::pair<OperatorSet, c10::Device (*)(INFER_PARAMS)>>
    deviceFuncInit{
        {creationOps, inferDeviceCreationOps},
        {combineOps, inferDeviceCombineOps},
    };

static bool initialized = false;
OperatorMap<c10::SymbolicShape (*)(INFER_PARAMS)> shapeFuncs;
OperatorMap<c10::ScalarType (*)(INFER_PARAMS)> dtypeFuncs;
OperatorMap<c10::Device (*)(INFER_PARAMS)> deviceFuncs;

void initTensorTypeFuncs() {
    if (initialized) return;
    for (auto &pair : shapeFuncInit) shapeFuncs.insert(pair.first, pair.second);
    for (auto &pair : dtypeFuncInit) dtypeFuncs.insert(pair.first, pair.second);
    for (auto &pair : deviceFuncInit)
        deviceFuncs.insert(pair.first, pair.second);
    initialized = true;
}

}  // namespace jit
}  // namespace torch
