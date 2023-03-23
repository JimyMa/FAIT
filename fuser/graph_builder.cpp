//
// Created by jimyma on 1/27/23.
//
#include "fuser/graph_builder.h"

#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/TensorGeometry.h>
#include <ATen/core/List.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/jit_type_base.h>
#include <ATen/core/stack.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <c10/util/string_utils.h>
#include <c10/util/variant.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/tensorexpr/graph_opt.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/loopnest_randomization.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>
#include <torch/csrc/jit/tensorexpr/types.h>

#include <memory>
#include <unordered_map>
#include <utility>

#include "fuser/codegen.h"
#include "passes/te_op.h"
#include "passes/tensor_ssa.h"
#include "tensorexpr/evaluate_output_shape.h"
#include "tensorexpr/functor_parallization.h"
#include "tensorexpr/parallel_for_equal_substitution.h"
#include "tensorexpr/tuple_expr.h"
#include "util/logging.h"

using namespace torch::jit::tensorexpr;

namespace torch {
namespace jit {

std::vector<int64_t> GraphBuilder::get_stride_by_shape(
    const std::vector<int64_t> shape) const {
  std::vector<int64_t> result;
  int64_t base = 1;
  for (int i = shape.size() - 1; i >= 0; i--) {
    result.insert(result.begin(), base);
    base *= shape[i];
  }
  return result;
}

std::vector<ExprHandle> GraphBuilder::get_stride_by_expr_dims(
    const std::vector<ExprHandle> expr_shape) const {
  std::vector<ExprHandle> result;
  ExprHandle base = LongImm::make(1);
  for (int i = expr_shape.size() - 1; i >= 0; i--) {
    result.insert(result.begin(), base);
    base = base * expr_shape[i];
  }
  return result;
}

Dtype GraphBuilder::get_scalar_type_by_value_type(TypePtr value_type) {
  std::unordered_map<TypeKind, ScalarType> type_map = {
      {TypeKind::FloatType, ScalarType::Float},
      {TypeKind::IntType, ScalarType::Int},
      {TypeKind::BoolType, ScalarType::Bool}};
  return ToDtype(type_map[value_type->kind()]);
}

VarHandle GraphBuilder::get_const_var_by_value(const Value* value) {
  Dtype scalar_type = get_scalar_type_by_value_type(value->type());
  return VarHandle(set_hash_name("Const"), scalar_type);
}

GraphBuilder::GraphBuilder(const Node* node)
    : graph_(node->g(attr::Subgraph)),
      refined_types_(node->tys(c10::tssa::input_refine_types)),
      is_parallelled_args_(node->is(c10::tssa::is_parallelled_args)),
      degree_(node->i(c10::tssa::parallel_degree)),
      is_parallel_map_(node->i(c10::tssa::is_parallel_map)) {
  compile();
}

std::vector<ArgValue> GraphBuilder::get_input_expr(Node* node) {
  std::vector<ArgValue> inputs_expr;
  for (auto input_ : node->inputs()) {
    if (input_->node()->kind() == prim::Constant) {
      auto val = toIValue(input_).value();
      if (val.isDouble()) {
        inputs_expr.emplace_back(val.toDouble());
      } else if (val.isInt()) {
        inputs_expr.emplace_back(val.toInt());
      } else if (val.isBool()) {
        inputs_expr.emplace_back(val.toBool());
      } else if (val.isNone()) {
        // This is just a placeholder so we don't throw.  None-handling
        // is operator-specific and should be handled properly in
        // the operator-specific lowering code.
        inputs_expr.emplace_back(ArgNone());
      } else if (val.isIntList()) {
        inputs_expr.emplace_back(val.toIntVector());
      } else if (val.isDevice()) {
        inputs_expr.emplace_back(ArgNone());
      } else {
        TORCH_CHECK(false, "Type ", *val.type(),
                    " not supported for constant value")
        throw unsupported_dtype(val.type()->annotation_str());
      }
    } else if (input_->type()->cast<TensorType>()) {
      inputs_expr.emplace_back(BufHandle(exprs_[input_].AsNode<Buf>()));
    } else {
      inputs_expr.emplace_back(VarHandle(exprs_[input_].AsNode<Var>()));
    }
  }
  return inputs_expr;
}

ExprPtr GraphBuilder::solveScalarInput(TypePtr type) {
  switch (type->kind()) {
    case TypeKind::TupleType: {
      std::vector<ExprPtr> elements;
      for (auto element_type : type->cast<TupleType>()->containedTypes()) {
        auto element = solveScalarInput(element_type);
        elements.emplace_back(element);
      }
      return Tuple::make(elements, ToDtype(ScalarType::Undefined)).node();
    }
    case TypeKind::FloatType: {
      VarHandle v(set_hash_name("InputVar"), kFloat);
      return v.node();
    }
    case TypeKind::BoolType: {
      VarHandle v(set_hash_name("InputVar"), kBool);
      return v.node();
    }
    case TypeKind::IntType: {
      VarHandle v(set_hash_name("InputVar"), kLong);
      return v.node();
    }
    default: {
      throw unsupported_dtype(type->repr_str());
      break;
    }
  }
}

static std::vector<VarHandle> getVarListByTupleNode(ExprPtr expr) {
  if (auto var_node = to<Var>(expr)) {
    return std::vector<VarHandle>({
        VarHandle(var_node),
    });
  } else if (auto tuple_node = to<Tuple>(expr)) {
    std::vector<VarHandle> result;
    for (auto element : tuple_node->elements()) {
      auto var_list = getVarListByTupleNode(element);
      result.insert(result.end(), var_list.begin(), var_list.end());
    }
    return result;
  } else {
    LONG_TAIL_ABORT("UNSUPPORT TYPE FOR getVarListByTupleNode: "
                    << toString(expr->dtype().scalar_type()));
  }
}

void GraphBuilder::solveInput(Value* input_) {
  switch (input_->type()->kind()) {
    case TypeKind::TensorType: {
      BufHandle input_buf(
          set_hash_name("InputBuf"), FunctorShapeMap_[input_],
          ToDtype(input_->type()->cast<TensorType>()->scalarType().value()));
      exprs_[input_] = ExprHandle(input_buf.node());
      FunctorInputArgs.emplace_back(input_buf);
      break;
    }
    case TypeKind::FloatType:
    case TypeKind::BoolType:
    case TypeKind::IntType: {
      auto expr = solveScalarInput(input_->type());
      exprs_[input_] = ExprHandle(expr);
      auto var = VarHandle(to<Var>(expr));
      FunctorInputArgs.emplace_back(var);
      break;
    }
    case TypeKind::TupleType: {
      auto expr = solveScalarInput(input_->type());
      exprs_[input_] = ExprHandle(expr);
      auto var_list = getVarListByTupleNode(expr);
      FunctorInputArgs.emplace_back(var_list);
      break;
    }
    default: {
      throw unsupported_dtype(input_->type()->repr_str());
      break;
    }
  }
}

void GraphBuilder::compile() {
  nInputs_ = graph_->inputs().size();
  nOutputs_ = graph_->outputs().size();

  // Step 0: find common shape and dyn shape
  // For Common shape, use it by LongImm.
  // For Dyn shape, use it by VarHandle.

  // Step 1: Bind inputs to buffers.
  auto block = alloc<torch::jit::tensorexpr::Block>(std::vector<StmtPtr>({}));

  // Get Shape VarHandle
  auto parallel_args_idx = 0;
  auto graph_args_idx = 0;
  for (auto is_paralleled_arg : is_parallelled_args_) {
    auto graph_arg = graph_->inputs()[graph_args_idx];
    graph_args_idx += 1;

    if (is_paralleled_arg) {
      auto type_arg = refined_types_[parallel_args_idx];
      parallel_args_idx += 1;
      if (auto tensor_type_arg = type_arg->cast<TensorType>()) {
        auto symbolic_dims = tensor_type_arg->symbolic_sizes();
        std::vector<ExprHandle> value_dims_expr;
        for (int i = 0; i < symbolic_dims.rank(); i++) {
          if (symbolic_dims[i].is_static()) {
            value_dims_expr.push_back(LongImm::make(symbolic_dims[i].value()));
          } else {
            auto dim_expr = VarHandle(set_hash_name("dim"), kLong);
            value_dims_expr.push_back(dim_expr);
            FunctorShapeVarArgs.emplace_back(dim_expr);
          }
        }
        FunctorShapeMap_[graph_arg] = value_dims_expr;
      }
    }
  }

  // Input Buffer
  LONG_TAIL_LOG_INFO("Input Buffer begin");
  for (auto input_ : graph_->inputs()) {
    // AT_ASSERT(input_->type()->cast<TensorType>(), "Parallel Functor Only
    // Tensor Type are Supported by now");
    // AT_ASSERT(input_->type()->cast<TensorType>()->scalarType().has_value(),
    // "ScalarType must be complete");
    solveInput(input_);
  }
  LONG_TAIL_LOG_INFO("Input Buffer end!!");
  // Step 2: Bind Node to Compute Op
  LONG_TAIL_LOG_INFO("Bind Node to Compute Op Begin");
  for (auto node : graph_->nodes()) {
    auto inputs_expr = get_input_expr(node);
    switch (node->kind()) {
      case prim::Constant: {
        auto output_value = node->output(0);
        auto const_var = get_const_var_by_value(output_value);
        exprs_[output_value] = ExprHandle(const_var.node());
      } break;

      case aten::size: {
        TORCH_CHECK(node->inputs().size() == 2);
        BufHandle self(exprs_.at(node->input(0)).AsNode<Buf>());
        auto rank = self.dims().size();
        auto dim = *constant_as<int64_t>(node->input(1));
        if (dim < 0) dim += rank;
        exprs_[node->output(0)] = self.dim(dim);
      } break;

      case prim::ListConstruct:
        break;
      case prim::TupleUnpack: {
        for (int i = 0; i < node->outputs().size(); i++) {
          auto output_value = node->output(i);
          exprs_[output_value] =
              ExprHandle(exprs_[node->input(0)].AsNode<Tuple>()->elements()[i]);
        }
        break;
      }
      default: {
        TORCH_CHECK(node->maybeSchema(), "Schema not found for node ", *node);
        LONG_TAIL_LOG_INFO("Process Node Shape " << node->schema()
                                                 << " Begin ...");
        Tensor output_tensor(nullptr, nullptr);
        std::vector<ExprHandle> outputShape;

        if (node->isMemberOf(identicalShapeOps)) {
          outputShape = computeIdenticalShape(node, exprs_);
        } else if (shapeFuncs.contains(*node->maybeOperator())) {
          outputShape =
              (*shapeFuncs.find(*node->maybeOperator()))(node, exprs_);
        } else {
          LONG_TAIL_ABORT("No nnc shape function to support node "
                          << *node->maybeSchema() << std::endl);
        }
        FunctorShapeMap_[node->output(0)] = outputShape;
        LONG_TAIL_LOG_INFO("Process Node Shape " << *node->maybeSchema()
                                                 << " End ...");

        LONG_TAIL_LOG_INFO("Process Node Compute Op: " << *node->maybeSchema()
                                                       << " Begin ...");

        if (customLoweringFuncs.contains(*node->maybeOperator())) {
          LONG_TAIL_LOG_INFO("custom lowering ...");
          output_tensor = (*customLoweringFuncs.find(*node->maybeOperator()))(
              node, exprs_, outputShape,
              node->output(0)
                  ->type()
                  ->cast<TensorType>()
                  ->scalarType()
                  .value());
        } else if (node->maybeSchema()) {
          NNCLoweringFunction lowering;
          lowering = getStandardLoweringFor(c10::toString(node->schema()));
          if (lowering) {
            LONG_TAIL_LOG_INFO("standard lowering ...");
            get_stride_by_expr_dims(outputShape);

            output_tensor = lowering(inputs_expr, outputShape,
                                     get_stride_by_expr_dims(outputShape),
                                     node->output(0)
                                         ->type()
                                         ->cast<TensorType>()
                                         ->scalarType()
                                         .value(),
                                     device_);
          } else {
            LONG_TAIL_ABORT("Cannot find compute op for "
                            << *node->maybeSchema());
          }
        } else {
          LONG_TAIL_ABORT("Cannot find compute op for "
                          << *node->maybeSchema());
        }
        if (output_tensor.buf())
          exprs_[node->output(0)] = ExprHandle(output_tensor.buf());
        block->append_stmt(output_tensor.stmt());
        LONG_TAIL_LOG_INFO("Process Node Compute Op: " << *node->maybeSchema()
                                                       << " End ...");
      }
    }
  }
  LONG_TAIL_LOG_INFO("Bind Node to Compute Op End");
  // Step 3: Register Output
  for (auto output : graph_->outputs()) {
    bufOutputs_.insert(exprs_[output].AsNode<Buf>());
    FunctorOutputBufArgs.emplace_back(BufHandle(exprs_[output].AsNode<Buf>()));
  }

  // Step 4: Functor Parallelization
  // CodeGen
  LoopNest l(block, bufOutputs_);
  LoopNest::sanitizeNames(l.root_stmt());
  {
    LONG_TAIL_LOG_INFO("Original Functor: ");
    LONG_TAIL_LOG_INFO(to_string(l.root_stmt()));
  }

  l.simplify();
  l.inlineIntermediateBufs(true);

  auto stmt_ = l.root_stmt();
  {
    LONG_TAIL_LOG_INFO("after compute inline: ");
    LONG_TAIL_LOG_INFO(to_string(stmt_));
  }

  // Step 2.2: Loop Binding
  for (auto buf : bufOutputs_) {
    std::vector<ForPtr> loops = l.getLoopStmtsFor(buf);
    if (loops.empty()) {
      // This happens when Buf is 0-dim
      continue;
    }
    ForPtr flattened = nullptr;
    LoopNest::flatten(loops, &flattened);
    assert(flattened);

    int loopLevels = -1;
    const int kDefaultLoopLevels = 2;

    loopLevels = (loopLevels > 0) ? loopLevels : kDefaultLoopLevels;

    int blockCount = -1;
    int blockSize = -1;

    ForPtr inner;
    const int kDefaultBlockSize = 512;
    blockSize = (blockSize > 0) ? blockSize : kDefaultBlockSize;
    LoopNest::splitWithMask(flattened, blockSize, &inner);
    flattened->set_gpu_block_index(0);
    inner->set_gpu_thread_index(0);
  }

  auto new_loop_axis = VarHandle("new_axis_i", kLong);
  stmt_ = alloc<For>(new_loop_axis.node(), LongImm::make(0).node(),
                     LongImm::make(degree_).node(), stmt_);
  static_to<For>(stmt_)->set_gpu_block_index(1);

  {
    LONG_TAIL_LOG_INFO("after loop binding: ");
    LONG_TAIL_LOG_INFO(to_string(stmt_));
  }
  for (auto functor_dim : FunctorShapeVarArgs) {
    std::vector<VarHandle> par_dims;
    for (int i = 0; i < degree_; i++) {
      auto par_dim = VarHandle(
          set_hash_name(functor_dim.AsNode<Var>()->name_hint()), kLong);
      par_dims.push_back(par_dim);
    }
    ShapeVarParallelFunctorMap[functor_dim.AsNode<Var>()] = par_dims;
  }
  // Input Value Replacement
  std::unordered_map<const Value*, BufPtr> input_buf;
  for (auto input : graph_->inputs()) {
    if (input->type()->cast<TensorType>()) {
      auto functor_buf = exprs_[input].AsNode<Buf>();
      std::vector<BufHandle> par_bufs;
      for (int i = 0; i < degree_; i++) {
        std::vector<ExprHandle> par_dims;
        for (auto value_dim_idx = 0;
             value_dim_idx <
             input->type()->cast<TensorType>()->symbolic_sizes().rank();
             value_dim_idx++) {
          auto value_dim = input->type()
                               ->cast<TensorType>()
                               ->symbolic_sizes()[value_dim_idx];
          if (value_dim.is_static()) {
            par_dims.push_back(LongImm::make(value_dim.value()));
          } else {
            auto functor_dim =
                FunctorShapeMap_[input][value_dim_idx].AsNode<Var>();
            par_dims.push_back(ShapeVarParallelFunctorMap[functor_dim][i]);
          }
        }

        auto par_buf = BufHandle(set_hash_name(functor_buf->name_hint()),
                                 ExprVectorToExprHandleVector(
                                     ExprHandleVectorToExprVector(par_dims)),
                                 functor_buf->dtype());
        par_bufs.push_back(par_buf);
      }
      LoadBufParallelFunctorMap[exprs_[input].AsNode<Buf>()] = par_bufs;
    } else if (input->type()->cast<TupleType>()) {
      auto functor_var_list =
          getVarListByTupleNode(exprs_[input].AsNode<Tuple>());
      for (auto functor_expr : functor_var_list) {
        auto functor_var = functor_expr.node();
        std::vector<VarHandle> par_vars;
        for (int i = 0; i < degree_; i++) {
          auto par_var = VarHandle(set_hash_name(functor_var->name_hint()),
                                   functor_var->dtype());
          par_vars.push_back(par_var);
        }
        LoadVarParallelFunctorMap[functor_var] = par_vars;
      }
    } else {
      auto functor_var = exprs_[input].AsNode<Var>();
      std::vector<VarHandle> par_vars;
      for (int i = 0; i < degree_; i++) {
        auto par_var = VarHandle(set_hash_name(functor_var->name_hint()),
                                 functor_var->dtype());
        par_vars.push_back(par_var);
      }
      LoadVarParallelFunctorMap[exprs_[input].AsNode<Var>()] = par_vars;
    }
  }

  stmt_ = FunctorParallization::parallel_functor_load(
      stmt_, degree_, new_loop_axis.node(), LoadBufParallelFunctorMap,
      LoadVarParallelFunctorMap);
  l.simplify();
  std::unordered_map<const Value*, BufPtr> output_buf;
  for (auto output : graph_->outputs()) {
    auto functor_buf = exprs_[output].AsNode<Buf>();

    std::vector<BufHandle> par_bufs;
    for (int i = 0; i < degree_; i++) {
      std::vector<ExprHandle> par_dims;
      for (int dim_idx = 0; dim_idx < functor_buf->dims().size(); dim_idx++) {
        auto functor_dim = ExprHandle(functor_buf->dim(dim_idx)).AsNode<Var>();
        if (!functor_dim) {
          par_dims.push_back(ExprHandle(functor_buf->dim(dim_idx)));
        } else {
          par_dims.push_back(
              ExprHandle(ShapeVarParallelFunctorMap[functor_dim][i]));
        }
      }

      auto par_buf = BufHandle(
          set_hash_name(functor_buf->name_hint()),
          ExprVectorToExprHandleVector(ExprHandleVectorToExprVector(par_dims)),
          functor_buf->dtype());
      par_bufs.push_back(par_buf);
    }
    StoreBufParallelFunctorMap[exprs_[output].AsNode<Buf>()] = par_bufs;
  }
  stmt_ = FunctorParallization::parallel_functor_store(
      stmt_, degree_, new_loop_axis.node(), StoreBufParallelFunctorMap);
  l.simplify();
  stmt_ = FunctorParallization::parallel_functor_shape(
      stmt_, degree_, new_loop_axis.node(), ShapeVarParallelFunctorMap);
  l.simplify();
  {
    LONG_TAIL_LOG_INFO("after parallization: ");
    LONG_TAIL_LOG_INFO(to_string(stmt_));
  }

  ParallelForEqualSubstitution::run(stmt_);
  l.simplify();
  {
    LONG_TAIL_LOG_INFO("after parallel for equal substitution: ");
    LONG_TAIL_LOG_INFO(to_string(stmt_));
  }

  l.prepareForCodegen();
  l.simplify();
  auto stmt = l.root_stmt();
  IRSimplifier::simplify(stmt);
  {
    LONG_TAIL_LOG_INFO("after pre codegen: ");
    LONG_TAIL_LOG_INFO(to_string(stmt_));
  }

  // input
  // buf
  for (auto input : FunctorInputArgs) {
    if (auto buf_input = c10::get_if<BufHandle>(&input)) {
      auto par_inputs = LoadBufParallelFunctorMap[buf_input->AsNode<Buf>()];
      for (auto par_input : par_inputs) {
        ParallelBufferArgs_.push_back(par_input);
      }
    } else if (auto var_input = c10::get_if<VarHandle>(&input)) {
      auto par_inputs = LoadVarParallelFunctorMap[var_input->AsNode<Var>()];
      for (auto par_input : par_inputs) {
        ParallelBufferArgs_.push_back(par_input);
      }
    } else if (auto var_list_input = c10::get_if<VarList>(&input)) {
      for (auto var_input : *var_list_input) {
        auto par_inputs = LoadVarParallelFunctorMap[var_input.AsNode<Var>()];
        for (auto par_input : par_inputs) {
          ParallelBufferArgs_.push_back(par_input);
        }
      }
    } else {
      throw unsupported_dtype();
    }
  }

  // shape
  for (auto dim : FunctorShapeVarArgs) {
    auto par_dims = ShapeVarParallelFunctorMap[dim.AsNode<Var>()];
    for (auto par_dim : par_dims) {
      ParallelBufferArgs_.push_back(par_dim);
    }
  }

  // output
  for (auto output : FunctorOutputBufArgs) {
    auto par_outputs = StoreBufParallelFunctorMap[output.AsNode<Buf>()];
    for (auto par_output : par_outputs) {
      ParallelBufferArgs_.push_back(par_output);
    }
  }

  codegen_ = CreateCodeGen("cuda_codegen", stmt_, ParallelBufferArgs_, device_);

  {
    LONG_TAIL_LOG_INFO("after codegen: ");
    LONG_TAIL_LOG_INFO(codegen_->getCodeText());
  }
}

void GraphBuilder::run(torch::jit::Stack& stack) const { runKernel(stack); }

std::vector<CodeGen::CallArg> GraphBuilder::prepareRunArgs(
    const at::ArrayRef<IValue>& inputs,
    std::vector<std::vector<at::Tensor>>& outputs) const {
  LONG_TAIL_LOG_INFO("solve input begin");
  std::vector<CodeGen::CallArg> runArgs;
  // TODO: with is_paralllel_args
  std::vector<CodeGen::CallArg> shape_args;
  LONG_TAIL_LOG_INFO("preparing input and shape call args ... ...");

  std::vector<CodeGen::CallArg> shapeRunArgs;
  std::unordered_map<VarPtr, int64_t> dim_map;
  for (int input_idx = 0; input_idx < inputs.size(); input_idx++) {
    Value* input_value = graph_->inputs()[input_idx];
    auto input = inputs[input_idx];
    auto functor_input = FunctorInputArgs[input_idx];
    if (input.isTensorList()) {
      auto list_input = input.toTensorList();
      auto functor_shape_expr = FunctorShapeMap_.at(input_value);

      for (int i = 0; i < degree_; i++) {
        auto tensor_input = list_input[i].get().toTensor().contiguous();
        runArgs.emplace_back(tensor_input.data_ptr());
      }

      auto tensor_type = refined_types_[input_idx]->cast<TensorType>();
      for (int64_t dim_idx = 0; dim_idx < tensor_type->sizes().size();
           dim_idx++) {
        if (!tensor_type->symbolic_sizes()[dim_idx].is_static()) {
          for (int i = 0; i < degree_; i++) {
            auto tensor_input = list_input[i].get().toTensor();
            auto dim_value = tensor_input.size(dim_idx);
            shapeRunArgs.emplace_back(dim_value);
            VarPtr functor_shape_var =
                functor_shape_expr[dim_idx].AsNode<Var>();
            dim_map[ShapeVarParallelFunctorMap.at(functor_shape_var)[i]
                        .node()] = dim_value;
          }
        }
      }
    } else if (input.isDoubleList()) {
      auto list_input = input.toDoubleList();
      for (int i = 0; i < degree_; i++) {
        auto value = list_input[i].get().toDouble();
        runArgs.emplace_back(value);
      }
    } else if (input.isIntList()) {
      auto list_input = input.toIntList();
      auto functor_input_var = *c10::get_if<VarHandle>(&functor_input);
      for (int i = 0; i < degree_; i++) {
        auto value = list_input[i].get().toInt();
        runArgs.emplace_back(value);
        dim_map[LoadVarParallelFunctorMap.at(functor_input_var.node())[i]
                    .node()] = value;
      }
    } else if (input.isBoolList()) {
      auto list_input = input.toBoolList();
      auto functor_input_var = *c10::get_if<VarHandle>(&functor_input);
      for (int i = 0; i < degree_; i++) {
        auto value = list_input[i].get().toBool();
        runArgs.emplace_back(value);
        dim_map[LoadVarParallelFunctorMap.at(functor_input_var.node())[i]
                    .node()] = int64_t(value);
      }
    } else if (input.isList() && input_value->type()->cast<TupleType>()) {
      auto degree = degree_;
      auto tuple_lens =
          input_value->type()->cast<TupleType>()->elements().size();
      auto functor_input_var_list = *c10::get_if<VarList>(&functor_input);
      for (int i = 0; i < tuple_lens; i++) {
        for (int j = 0; j < degree_; j++) {
          auto value =
              input.toList()[j].get().toTuple().get()->elements()[i].toInt();
          runArgs.emplace_back(value);
          dim_map[LoadVarParallelFunctorMap
                      .at(functor_input_var_list[i].node())[j]
                      .node()] = value;
        }
      }
    } else if (input.isTensor()) {
      auto tensor_input = input.toTensor().contiguous();
      auto functor_shape_expr = FunctorShapeMap_.at(input_value);

      std::vector<CodeGen::CallArg> shape_args_per_degree;
      runArgs.emplace_back(tensor_input.data_ptr());
      auto tensor_type = refined_types_[input_idx]->cast<TensorType>();
      for (int64_t dim_idx = 0; dim_idx < tensor_type->sizes().size();
           dim_idx++) {
        if (!tensor_type->symbolic_sizes()[dim_idx].is_static()) {
          VarPtr functor_shape_var = functor_shape_expr[dim_idx].AsNode<Var>();
          dim_map[ShapeVarParallelFunctorMap.at(functor_shape_var)[0].node()] =
              tensor_input.size(dim_idx);
          shapeRunArgs.emplace_back(tensor_input.size(dim_idx));
        }
      }
    } else if (input.isDouble()) {
      auto double_input = input.toDouble();
      for (int i = 0; i < degree_; i++) {
        runArgs.emplace_back(double_input);
      }
    } else {
      TORCH_CHECK(false, "Unsupported input type: ", input.tagKind());
    }
  }
  runArgs.insert(runArgs.end(), shapeRunArgs.begin(), shapeRunArgs.end());

  LONG_TAIL_LOG_INFO("preparing input and shape call args DONE!!!");

  for (int i = 0; i < graph_->outputs().size(); i++) {
    auto output_value = graph_->outputs()[i];
    std::vector<at::Tensor> list_output;
    for (int j = 0; j < degree_; j++) {
      // bufOutputs_
      std::vector<int64_t> output_shape;
      auto output_dims_expr = StoreBufParallelFunctorMap
                                  .at(exprs_.at(output_value).AsNode<Buf>())[j]
                                  .dims();
      for (auto output_dim_expr : output_dims_expr) {
        auto dim = EvaluateOutputShape::run(output_dim_expr.node(), dim_map, j);
        output_shape.push_back(dim);
      }
      auto output_tensor = codegen_->empty_strided(
          output_shape, get_stride_by_shape(output_shape),
          *output_value->type()->cast<TensorType>()->scalarType(),
          c10::kStrided, device_, false);
      list_output.emplace_back(output_tensor);
      runArgs.emplace_back(output_tensor.data_ptr());
    }
    outputs.push_back(list_output);
  }
  LONG_TAIL_LOG_INFO("solve input done");
  return runArgs;
}

void GraphBuilder::runKernel(Stack& stack) const {
  LONG_TAIL_LOG_INFO("run kernel begin: ");
  auto inputs = last(stack, nInputs_);
  std::vector<std::vector<at::Tensor>> outputs;
  LONG_TAIL_LOG_INFO("Preparing call args ... ... ");
  std::vector<CodeGen::CallArg> runArgs = prepareRunArgs(inputs, outputs);
  {
    LONG_TAIL_LOG_INFO("Preparing call args DONE!!! ");
    LONG_TAIL_LOG_INFO("run kernel call begin ... ");
  }

  codegen_->call(runArgs);
  LONG_TAIL_LOG_INFO("run kernel call end ...");

  drop(stack, nInputs_);

  for (auto& o : outputs) {
    if (is_parallel_map_) {
      push_one(stack, std::move(at::List<at::Tensor>(o)));
    } else {
      push_one(stack, std::move(o[0]));
    }
  }
}

}  // namespace jit
}  // namespace torch
