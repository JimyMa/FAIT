#include <functional>

#include "refine_types.h"
#include "type_utils.h"

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
};

static void updateDtypeFromArgs(Node *node, const FunctionSchema &schema,
                                c10::ScalarType &dtype) {
  // Check if there is source tensor (`self`)
  auto selfIdx = schema.argumentIndexWithName("self");
  if (selfIdx)
    dtype = *node->input(*selfIdx)->type()->cast<TensorType>()->scalarType();

  // Check if there is target tensor (`other`)
  auto otherIdx = schema.argumentIndexWithName("other");
  if (otherIdx)
    dtype = *node->input(*otherIdx)->type()->cast<TensorType>()->scalarType();

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

static OperatorSet tensorOps{
    "aten::tensor.float(float t, *, ScalarType? dtype=None, Device? "
    "device=None, bool requires_grad=False) -> Tensor",
    "aten::tensor.int(int t, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
    "aten::tensor.bool(bool t, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
    "aten::tensor(t[] data, *, ScalarType? dtype=None, Device? device=None, "
    "bool requires_grad=False) -> Tensor",
};

static c10::SymbolicShape inferShapeTensorOps(INFER_PARAMS) {
  auto value = node->input(0);
  auto type = value->type();
  if (value->type()->kind() == TypeKind::ListType) {
    auto len = getListLen(value, refinedTypes);
    if (len) {
      return c10::IntArrayRef({int64_t(*len)});
    } else
      return getRankedShape(1);
  } else {
    return getRankedShape(0);
  }
}

static std::unordered_map<TypeKind, c10::ScalarType> typeKindsToScalarTypes{
    {TypeKind::FloatType, c10::kFloat},
    {TypeKind::IntType, c10::kLong},
    {TypeKind::BoolType, c10::kBool},
};

static c10::ScalarType inferDtypeTensorOps(INFER_PARAMS) {
  auto value = node->input(0);
  auto type = value->type();
  auto kind = type->kind();
  if (typeKindsToScalarTypes.count(kind))
    return typeKindsToScalarTypes[kind];
  else if (kind == TypeKind::ListType) {
    auto elemTy = type->cast<ListType>()->getElementType();
    TORCH_CHECK(typeKindsToScalarTypes.count(elemTy->kind()));
    return typeKindsToScalarTypes[elemTy->kind()];
  } else {
    throw typeError("Cannot infer data type for input %", value->debugName(),
                    " of `aten::tensor`", c10::get_backtrace());
  }
}

static OperatorSet fillOps{
    "aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? "
    "layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
};

static c10::SymbolicShape inferShapeFillOps(INFER_PARAMS) {
  auto size = getIntList(node->input(0));
  if (!size) return {};
  return *size;
}

static OperatorSet bcastOps{
    "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
    "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
    "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::div.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::minimum(Tensor self, Tensor other) -> Tensor",
    "aten::maximum(Tensor self, Tensor other) -> Tensor",
    "aten::eq.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::ne.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::lt.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::gt.Tensor(Tensor self, Tensor other) -> Tensor",
    "aten::ge.Tensor(Tensor self, Tensor other) -> Tensor",
};

static ShapeDim bcastDim(const ShapeDim &lhs, const ShapeDim &rhs) {
  // Not clear if neither dimension is known
  if (!lhs && !rhs) return c10::nullopt;

  // Select the larger dimension size if both are known
  if (lhs && rhs) return std::max(*lhs, *rhs);

  // Infer if only one dimension is known
  if (lhs) {
    if (*lhs > 1)
      return lhs;
    else
      return c10::nullopt;
  } else {
    if (*rhs > 1)
      return rhs;
    else
      return c10::nullopt;
  }

  return c10::nullopt;
}

static c10::SymbolicShape inferShapeBcastOps(INFER_PARAMS) {
  // Process input shapes
  auto lShape = getShape(node->input(0)->type()),
       rShape = getShape(node->input(1)->type());
  if (!lShape || !rShape) return {};
  int64_t lRank = lShape->size(), rRank = rShape->size();

  // Infer output shape
  auto outRank = std::max(lRank, rRank);
  ShapeVec outShape(size_t(outRank), c10::nullopt);
  for (auto i : c10::irange(outRank)) {
    auto lIdx = lRank - 1 - i, rIdx = rRank - 1 - i;
    ShapeDim outDim;
    if (lIdx < 0)
      outDim = rShape->at(rIdx);
    else if (rIdx < 0)
      outDim = lShape->at(lIdx);
    else
      outDim = bcastDim(lShape->at(lIdx), rShape->at(rIdx));
    outShape[outRank - 1 - i] = outDim;
  }

  return outShape;
}

static OperatorSet selectOp{
    "aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)",
};

static c10::SymbolicShape inferShapeSelectOp(INFER_PARAMS) {
  // Process argument
  auto inShape = getShape(node->input(0)->type());
  if (!inShape) return {};
  auto rank = inShape->size();
  auto dimIVal = toIValue(node->input(1));
  if (!dimIVal) return getRankedShape(rank - 1);
  auto dim = dimIVal->toInt();
  if (dim < 0) dim += rank;

  // Infer output shape
  inShape->erase(inShape->begin() + dim);
  return *inShape;
}

static c10::optional<int64_t> refineDimSizeIndex(
    Value *indexValue, const c10::optional<int64_t> &defaultIfNone) {
  c10::optional<int64_t> index;
  auto ival = toIValue(indexValue);
  if (!ival) return c10::nullopt;
  if (ival->isNone())
    index = defaultIfNone;
  else if (ival->isInt())
    index = ival->toInt();
  return index;
}

static OperatorSet sliceOp{
    "aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? "
    "end=None, SymInt step=1) -> Tensor(a)",
};

static c10::SymbolicShape inferShapeSliceOp(INFER_PARAMS) {
  // Process argument
  auto inShape = getShape(node->input(0)->type());
  if (!inShape) return {};
  auto rank = inShape->size();
  auto dimIVal = toIValue(node->input(1));
  if (!dimIVal) return getRankedShape(rank);
  auto dim = dimIVal->toInt();
  if (dim < 0) dim += rank;

  // Process dimension range
  auto dimSize = inShape->at(dim);
  auto start = refineDimSizeIndex(node->input(2), 0);
  auto end = refineDimSizeIndex(node->input(3), dimSize);
  if (dimSize) {
    if (start && *start < 0) *start += *dimSize;
    if (end) {
      if (*end < 0)
        *end += *dimSize;
      else
        *end = std::min(*end, *dimSize);
    }
  }
  auto step = refineDimSizeIndex(node->input(4), 1);
  auto outDimSize = tryApply<int64_t>(
      [](int64_t start, int64_t end, int64_t step) {
        return (end - start - 1) / step + 1;
      },
      start, end, step);

  // Compute output shape
  ShapeVec outShape;
  for (auto i : c10::irange(rank)) {
    ShapeDim size;
    if (i == dim)
      size = outDimSize;
    else
      size = inShape->at(i);
    outShape.push_back(size);
  }

  return outShape;
}

static OperatorSet squeezeOp{
    "aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)",
};

static c10::SymbolicShape inferShapeSqueezeOp(INFER_PARAMS) {
  // Process arguments
  auto inShape = getShape(node->input(0)->type());
  if (!inShape) return {};
  auto rank = inShape->size();
  auto dimIVal = toIValue(node->input(1));
  if (!dimIVal) return {};
  auto dim = dimIVal->toInt();
  if (dim < 0) dim += rank;

  // Remove dimension from shape
  auto dimSize = inShape->at(dim);
  if (!dimSize) return {};
  if (*dimSize == 1) inShape->erase(inShape->begin() + dim);
  return *inShape;
}

static OperatorSet unsqueezeOp{
    "aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)",
};

static c10::SymbolicShape inferShapeUnsqueezeOp(INFER_PARAMS) {
  // Process arguments
  auto inShape = getShape(node->input(0)->type());
  if (!inShape) return {};
  auto rank = inShape->size();
  auto dimIVal = toIValue(node->input(1));
  if (!dimIVal) return getRankedShape(rank + 1);
  auto dim = dimIVal->toInt();
  if (dim < 0) dim += rank + 1;

  // Insert dimension to shape
  inShape->insert(inShape->begin() + dim, 1);
  return *inShape;
}

static OperatorSet reshapeOps{
    "aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)",
    "aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)",
};

static c10::SymbolicShape inferShapeReshapeOps(INFER_PARAMS) {
  auto shape = getIntList(node->input(1));
  if (shape)
    return *shape;
  else
    return {};
}

static OperatorSet permuteOp{
    "aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)",
};

static c10::SymbolicShape inferShapePermuteOp(INFER_PARAMS) {
  // Get self shape and dims
  auto inShape = getShape(node->input(0)->type());
  if (!inShape) return {};
  auto dims = getIntList(node->input(1));
  if (!dims) return getRankedShape(inShape->size());
  TORCH_CHECK(inShape->size() == dims->size());

  // Permute dimensions
  ShapeVec outShape;
  for (auto i : c10::irange(dims->size())) {
    auto dimIdx = dims->at(i);
    ShapeDim shapeDim;
    if (dimIdx) shapeDim = inShape->at(*dimIdx);
    outShape.push_back(shapeDim);
  }

  return outShape;
}

static OperatorSet expandOp{
    "aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> "
    "Tensor(a)",
};

static c10::SymbolicShape inferShapeExpandOp(INFER_PARAMS) {
  // Get shape and expand sizes
  auto inShape = getShape(node->input(0)->type());
  if (!inShape) return {};
  auto sizes = getIntList(node->input(1));
  if (!sizes) return {};
  auto inRank = int64_t(inShape->size()), sizeLen = int64_t(sizes->size());

  // Compute output shape
  auto outRank = std::max(inRank, sizeLen);
  ShapeVec outShape(outRank, c10::nullopt);
  for (auto i : c10::irange(outRank)) {
    auto inIdx = inRank - 1 - i, sizeIdx = sizeLen - 1 - i,
         outIdx = outRank - 1 - i;
    if (inIdx < 0) {
      outShape[outIdx] = sizes->at(sizeIdx);
      continue;
    }
    if (sizeIdx < 0) {
      outShape[outIdx] = inShape->at(inIdx);
      continue;
    }
    outShape[outIdx] = tryApply<int64_t>(
        [](int64_t inDim, int64_t sizeDim) {
          if (sizeDim < 0)
            return inDim;
          else
            return std::max(inDim, sizeDim);
        },
        inShape->at(inIdx), sizes->at(sizeIdx));
  }

  return outShape;
}

static OperatorSet repeatOp{
    "aten::repeat(Tensor self, SymInt[] repeats) -> Tensor"};

static c10::SymbolicShape inferShapeRepeatOp(INFER_PARAMS) {
  // Get shape and repeats
  auto inShape = getShape(node->input(0)->type());
  if (!inShape) return {};
  auto repeats = getIntList(node->input(1));
  if (!repeats) return {};
  auto inRank = int64_t(inShape->size()), repeatLen = int64_t(repeats->size());

  // Compute output shape
  auto outRank = std::max(inRank, repeatLen);
  ShapeVec outShape(outRank, c10::nullopt);
  for (auto i : c10::irange(outRank)) {
    auto inIdx = inRank - 1 - i, repIdx = repeatLen - 1 - i,
         outIdx = outRank - 1 - i;
    if (inIdx < 0) {
      outShape[outIdx] = repeats->at(repIdx);
      continue;
    }
    if (repIdx < 0) {
      outShape[outIdx] = inShape->at(inIdx);
      continue;
    }
    outShape[outIdx] = tryApply<int64_t>(
        std::multiplies<int64_t>(), inShape->at(inIdx), repeats->at(repIdx));
  }

  return outShape;
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

static OperatorSet catOp{
    "aten::cat(Tensor[] tensors, int dim=0) -> Tensor",
};

static c10::SymbolicShape inferShapeCatOp(INFER_PARAMS) {
  // Decide input tensor ranks
  auto listTy = getRefinedType(node->input(0), refinedTypes);
  auto rank = accumAttrFromElements<size_t>(listTy, getRank);
  if (!rank) return {};

  // Determine insert dimension
  auto dimIVal = toIValue(node->input(1));
  if (!dimIVal) return getRankedShape(*rank);
  auto dim = dimIVal->toInt();
  if (dim < 0) dim += *rank;

  // Propagate outout shape
  auto defaultShape = c10::VaryingShape<int64_t>(*rank).sizes();
  auto initShape = defaultShape;
  initShape->at(dim) = 0;
  auto shape = *accumAttrFromElements(
      listTy, getShape,
      [&](c10::optional<ShapeVec> &&accum,
          c10::optional<ShapeVec> &&newShape) -> c10::optional<ShapeVec> {
        if (!newShape) newShape = defaultShape;
        TORCH_CHECK(accum->size() == newShape->size());
        for (auto i : c10::irange(accum->size())) {
          const auto &accumDim = accum->at(i), &newDim = newShape->at(i);
          ShapeDim outDim;
          if (i == dim)
            outDim = tryApply<int64_t>(std::plus<int64_t>(), accumDim, newDim);
          else
            outDim = joinOpt(accumDim, newDim);
          accum->at(i) = outDim;
        }
        return std::move(accum);
      },
      initShape);

  return shape;
}

static OperatorSet stackOp{
    "aten::stack(Tensor[] tensors, int dim=0) -> Tensor",
};

static c10::SymbolicShape inferShapeStackOp(INFER_PARAMS) {
  // Decide input tensor ranks
  auto listTy = getRefinedType(node->input(0), refinedTypes);
  auto rank = accumAttrFromElements<size_t>(listTy, getRank);
  if (!rank) return {};

  // Determine insert dimension
  auto dimIVal = toIValue(node->input(1));
  if (!dimIVal) return getRankedShape(*rank + 1);
  auto dim = dimIVal->toInt();
  if (dim < 0) dim += (*rank + 1);

  // Propagate outout shape
  auto defaultShape = c10::VaryingShape<int64_t>(*rank).sizes();
  auto shape = *accumAttrFromElements(
      listTy, getShape,
      [&](c10::optional<ShapeVec> &&accum,
          c10::optional<ShapeVec> &&newShape) -> c10::optional<ShapeVec> {
        if (!newShape) newShape = defaultShape;
        TORCH_CHECK(accum->size() == newShape->size());
        for (auto i : c10::irange(accum->size()))
          accum->at(i) = joinOpt(accum->at(i), newShape->at(i));
        return std::move(accum);
      },
      defaultShape);

  // Insert axis to the group
  auto numTensors = mapOpt<int64_t>(getListLen(node->input(0), refinedTypes),
                                    [](size_t i) { return int64_t(i); });
  shape.insert(shape.begin() + dim, numTensors);

  return shape;
}

static OperatorSet indexOp{
    "aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor",
};

static c10::SymbolicShape inferShapeIndexOp(INFER_PARAMS) {
  // Get tensor shapes
  auto selfShape = getShape(node->input(0)->type());
  if (!selfShape) return {};
  // only support advanced indexing with exactly one tensor in `indices`
  TORCH_CHECK(*getListLen(node->input(1), refinedTypes) == 1);
  auto indexTy = getElementType(getRefinedType(node->input(1), refinedTypes), 0)
                     ->cast<TensorType>();
  if (!indexTy->dim()) return {};
  auto indexRank = *indexTy->dim();

  // Infer shape according to index data type
  auto indexDtype = indexTy->scalarType();
  TORCH_CHECK(indexDtype.has_value());
  switch (*indexDtype) {
    case c10::kBool: {
      selfShape->erase(selfShape->begin(), selfShape->begin() + indexRank - 1);
      selfShape->at(0) = c10::nullopt;
    } break;

    case c10::kLong: {
      TORCH_CHECK(indexRank == 1);
      selfShape->at(0) = c10::nullopt;
    } break;

    default: {
      TORCH_CHECK(false);
    }
  }
  return *selfShape;
}

static OperatorSet nonzeroOp{
    "aten::nonzero(Tensor self) -> Tensor",
};

static c10::SymbolicShape inferShapeNonzeroOp(INFER_PARAMS) {
  auto inRank = mapOpt<int64_t>(getRank(node->input(0)->type()),
                                [](size_t r) { return int64_t(r); });
  return ShapeVec{c10::nullopt, inRank};
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
    "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
    "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::div.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::eq.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::ne.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::gt.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::ge.Scalar(Tensor self, Scalar other) -> Tensor",
    "aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
    "aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor "
    "values, Tensor indices)",
};

static c10::SymbolicShape passSameShape(INFER_PARAMS) {
  return node->input(0)->type()->cast<TensorType>()->symbolic_sizes();
}

static OperatorSet rankZeroOps{
    "aten::max(Tensor self) -> Tensor",
    "aten::min(Tensor self) -> Tensor",
    "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor",
    "aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor",
    "aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor",
};

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

static void handleDtypeSort(INFER_PARAMS) {
  auto dtype = *node->input(0)->type()->cast<TensorType>()->scalarType();
  node->output(0)->setType(
      node->output(0)->type()->cast<TensorType>()->withScalarType(dtype));
  node->output(1)->setType(
      node->output(1)->type()->cast<TensorType>()->withScalarType(c10::kLong));
}

static std::initializer_list<
    std::pair<OperatorSet, c10::SymbolicShape (*)(INFER_PARAMS)>>
    shapeFuncInit{
        {tensorOps, inferShapeTensorOps},
        {fillOps, inferShapeFillOps},
        {bcastOps, inferShapeBcastOps},
        {selectOp, inferShapeSelectOp},
        {sliceOp, inferShapeSliceOp},
        {squeezeOp, inferShapeSqueezeOp},
        {unsqueezeOp, inferShapeUnsqueezeOp},
        {reshapeOps, inferShapeReshapeOps},
        {permuteOp, inferShapePermuteOp},
        {expandOp, inferShapeExpandOp},
        {repeatOp, inferShapeRepeatOp},
        {catOp, inferShapeCatOp},
        {stackOp, inferShapeStackOp},
        {indexOp, inferShapeIndexOp},
        {nonzeroOp, inferShapeNonzeroOp},
        {sameShapeOps, passSameShape},
        {rankZeroOps, [](INFER_PARAMS) { return getRankedShape(0); }},
        {rankOneOps, [](INFER_PARAMS) { return getRankedShape(1); }},
    };

static std::initializer_list<
    std::pair<OperatorSet, c10::ScalarType (*)(INFER_PARAMS)>>
    dtypeFuncInit{
        {convertOrFillOps, inferDtypeConvertOrFillOps},
        {tensorOps, inferDtypeTensorOps},
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

static std::initializer_list<std::pair<OperatorSet, void (*)(INFER_PARAMS)>>
    specialDtypeHandlerInit{
        {{"aten::sort(Tensor self, int dim=-1, bool descending=False) -> "
          "(Tensor values, Tensor indices)"},
         handleDtypeSort},
    };

static bool initialized = false;
OperatorMap<c10::SymbolicShape (*)(INFER_PARAMS)> shapeFuncs;
OperatorMap<c10::ScalarType (*)(INFER_PARAMS)> dtypeFuncs;
OperatorMap<c10::Device (*)(INFER_PARAMS)> deviceFuncs;
OperatorMap<void (*)(Node *, ValueTypeMap &)> specialDtypeHandlers;

void initTensorTypeFuncs() {
  if (initialized) return;
  for (auto &pair : shapeFuncInit) shapeFuncs.insert(pair.first, pair.second);
  for (auto &pair : dtypeFuncInit) dtypeFuncs.insert(pair.first, pair.second);
  for (auto &pair : deviceFuncInit) deviceFuncs.insert(pair.first, pair.second);
  for (auto &pair : specialDtypeHandlerInit)
    specialDtypeHandlers.insert(pair.first, pair.second);
  initialized = true;
}

}  // namespace jit
}  // namespace torch
