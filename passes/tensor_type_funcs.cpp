#include "refine_types.h"

namespace torch {
namespace jit {

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

static c10::Device inferDeviceCreationOps(Node *node, ValueTypeMap &) {
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
        auto deviceVal = node->input(*deviceIdx);
        auto ival = toIValue(node->input(*deviceIdx));
        if (ival && ival->isDevice()) device = (*ival).toDevice();
    }

    return device;
}

static OperatorSet combineOps{
    "aten::cat(Tensor[] tensors, int dim=0) -> Tensor",
    "aten::stack(Tensor[] tensors, int dim=0) -> Tensor",
};

static c10::Device inferDeviceCombineOps(Node *node,
                                         ValueTypeMap &refinedTypes) {
    auto listTy = refinedTypes.at(node->input(0));
    auto tensorTy = listTy->containedType(0)->cast<TensorType>();
    return *tensorTy->device();
}

static bool initialized = false;
OperatorMap<c10::SymbolicShape (*)(Node *, ValueTypeMap &)> shapeFuncs;
OperatorMap<c10::ScalarType (*)(Node *, ValueTypeMap &)> dtypeFuncs;
OperatorMap<c10::Device (*)(Node *, ValueTypeMap &)> deviceFuncs;

void initTensorTypeFuncs() {
    if (initialized) return;

    /* Shape functions */

    /* Dtype functions */

    /* Device functions */
    deviceFuncs.insert(creationOps, inferDeviceCreationOps);
    deviceFuncs.insert(combineOps, inferDeviceCombineOps);

    initialized = true;
}

}  // namespace jit
}  // namespace torch
