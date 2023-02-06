//
// Created by jimyma on 2/1/23.
//

//
// Created by jimyma on 1/29/23.
//

#include "ATen/Context.h"

#include "torch/csrc/jit/tensorexpr/analysis.h"
#include "torch/csrc/jit/tensorexpr/codegen.h"
#include "torch/csrc/jit/tensorexpr/expr.h"
#include "torch/csrc/jit/tensorexpr/ir_simplifier.h"
#include "torch/csrc/jit/tensorexpr/loopnest.h"

#include "tensorexpr/functor_parallization.h"

using namespace torch::jit::tensorexpr;

/*
pytorch hot spot code

def func(a: List[torch.Tensor], b: torch.Tensor)
  e_list = []
  for (a in a_list) {
    c = a + b
    d = c + b
    e_list.append(d[..., 0] + scalar_0)
  }
  return e_list
*/

int main() {
  at::globalContext().lazyInitCUDA();
  std::vector<CodeGen::BufferArg> bufferArgs_;
  std::unordered_set<BufPtr> bufOutputs_;

  // Functor Define
  // Shape Define
  auto N = VarHandle("dyn_shape_axes", kLong);
  // Input Define
  BufHandle a_buf("a_buf", {N, N}, kDouble);
  BufHandle b_buf("b_buf", {N, N}, kDouble);
  VarHandle scalar_0("scalar_0", kDouble);

  Tensor scalar_0_tensor(nullptr, nullptr);

  // Output Define
  Tensor c_tensor(nullptr, nullptr);
  Tensor d_tensor(nullptr, nullptr);
  Tensor e_tensor(nullptr, nullptr);

  // Compute Op Define
  c_tensor = Compute(
          "c_buf",
          {N, N},
          [&](const std::vector<VarHandle>& axes) {
              return a_buf.load(axes[0], axes[1]) + b_buf.load(axes[0], axes[1]);
          });

  d_tensor = Compute(
          "d_buf",
          {N, N},
          [&](const std::vector<VarHandle>& axes) {
              return b_buf.load(axes[0], axes[1]) + c_tensor.load(axes[0], axes[1]);
          });

  e_tensor = Compute(
          "e_buf",
          {N},
          [&](const std::vector<VarHandle>& axes) {
              return d_tensor.load(axes[0], 0) + scalar_0;
          });

  // Compute Op to Stmt
  auto block = alloc<Block>(std::vector<StmtPtr>({}));
  if (scalar_0_tensor.stmt())
    block->append_stmt(scalar_0_tensor.stmt());
  if (c_tensor.stmt())
    block->append_stmt(c_tensor.stmt());
  if (d_tensor.stmt())
    block->append_stmt(d_tensor.stmt());
  if (e_tensor.stmt())
    block->append_stmt(e_tensor.stmt());

  // Set Statement output
  bufOutputs_.insert(e_tensor.buf());

  // Loop Schedule
  LoopNest l(block, bufOutputs_);
  LoopNest::sanitizeNames(l.root_stmt());

  // Simplify
  l.simplify();

  // Compute Inline Begin
  l.inlineIntermediateBufs(/*allow_duplicated_work=*/true);
  l.optimizeConditionals();
  auto stmt_ = l.root_stmt();
  std::cout << "Inlined Statement... " << std::endl;
  std::cout << to_string(stmt_) << std::endl;
  // Compute Inline End

  // Functor Loop Binding
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
  std::cout << "Loop Binding: "  << std::endl;
  std::cout << to_string(l.root_stmt()) << std::endl;
  // Functor Parallelization
  // Add a new loop
  auto new_loop_axis = VarHandle("new_axis_i", kLong);
  stmt_ = alloc<For>(new_loop_axis.node(),
                     LongImm::make(0).node(),
                     LongImm::make(3).node(),
                     stmt_);

  // Change Functor Load / Store to Parallization Load / Store
  VarHandle N_0("dyn_shape_axes_0", kLong);
  VarHandle N_1("dyn_shape_axes_1", kLong);
  VarHandle N_2("dyn_shape_axes_2", kLong);
  BufHandle a_0_buf("a_0_tensor", {N_0, N_0}, kDouble);
  BufHandle a_1_buf("a_1_tensor", {N_1, N_1}, kDouble);
  BufHandle a_2_buf("a_2_tensor", {N_2, N_2}, kDouble);

  BufHandle b_0_buf("b_0_tensor", {N_0, N_0}, kDouble);
  BufHandle b_1_buf("b_1_tensor", {N_1, N_1}, kDouble);
  BufHandle b_2_buf("b_2_tensor", {N_2, N_2}, kDouble);

  BufHandle e_0_tensor("e_0_tensor", {N_0, N_0}, kDouble);
  BufHandle e_1_tensor("e_1_tensor", {N_1, N_1}, kDouble);
  BufHandle e_2_tensor("e_2_tensor", {N_2, N_2}, kDouble);
  stmt_ = FunctorParallization::parallel_functor_load(stmt_, 3, new_loop_axis.node(),
                                                      {
                                                              {a_buf.node(), {a_0_buf, a_1_buf, a_2_buf}},
                                                              {b_buf.node(), {b_0_buf, b_1_buf, b_2_buf}}
                                                      });

  stmt_ = FunctorParallization::parallel_functor_store(stmt_, 3, new_loop_axis.node(),
                                                       {
                                                               {e_tensor.buf(), {e_0_tensor, e_1_tensor, e_2_tensor}}
                                                       });
  stmt_ = FunctorParallization::parallel_functor_shape(stmt_, 3, new_loop_axis.node(),
                                                       {
                                                               {N.node(), {N_0, N_1, N_2}}
                                                       });
  static_to<For>(stmt_)->set_gpu_block_index(1);
  std::cout << to_string(stmt_) << std::endl;

  // CodeGen
  l.prepareForCodegen();
  l.simplify();

  auto stmt = l.root_stmt();
  IRSimplifier::simplify(stmt);

  // Kernel Arguments
  bufferArgs_.emplace_back(a_0_buf);
  bufferArgs_.emplace_back(a_1_buf);
  bufferArgs_.emplace_back(a_2_buf);
  bufferArgs_.emplace_back(b_0_buf);
  bufferArgs_.emplace_back(b_1_buf);
  bufferArgs_.emplace_back(b_2_buf);
  bufferArgs_.emplace_back(scalar_0);
  bufferArgs_.emplace_back(N_0);
  bufferArgs_.emplace_back(N_1);
  bufferArgs_.emplace_back(N_2);
  bufferArgs_.emplace_back(e_0_tensor);
  bufferArgs_.emplace_back(e_1_tensor);
  bufferArgs_.emplace_back(e_2_tensor);

  // NNC CodeGen
  auto codegen_ = CreateCodeGen(
          "cuda_codegen",
          stmt_,
          bufferArgs_,
          at::kCUDA);

  std::cout << codegen_->getCodeText() << std::endl;

  int64_t n_0_runtime = 32l;
  int64_t n_1_runtime = 16l;
  int64_t n_2_runtime = 8l;


  // PyTorch Runtime
  // Inputs
  auto a_0_runtime = at::ones({n_0_runtime, n_0_runtime}, at::kDouble).cuda();
  auto a_1_runtime = at::ones({n_1_runtime, n_1_runtime}, at::kDouble).cuda() * 2.0;
  auto a_2_runtime = at::ones({n_2_runtime, n_2_runtime}, at::kDouble).cuda() * 3.0;

  auto b_0_runtime = at::ones({n_0_runtime, n_0_runtime}, at::kDouble).cuda();
  auto b_1_runtime = at::ones({n_1_runtime, n_1_runtime}, at::kDouble).cuda() * 2.0;
  auto b_2_runtime = at::ones({n_2_runtime, n_2_runtime}, at::kDouble).cuda() * 3.0;

  auto scalar_0_runtime = 2.0;
  std::vector<c10::IValue> inputs = {a_0_runtime,
                                     a_1_runtime,
                                     a_2_runtime,
                                     b_0_runtime,
                                     b_1_runtime,
                                     b_2_runtime,
                                     scalar_0_runtime,
                                     n_0_runtime,
                                     n_1_runtime,
                                     n_2_runtime};

  // Outputs
  auto e_0_runtime = codegen_->empty_strided(
          {n_0_runtime, },
          {1, },
          c10::kDouble,
          c10::kStrided,
          c10::kCUDA,
          false);

  auto e_1_runtime = codegen_->empty_strided(
          {n_1_runtime, },
          {1, },
          c10::kDouble,
          c10::kStrided,
          c10::kCUDA,
          false);

  auto e_2_runtime = codegen_->empty_strided(
          {n_2_runtime, },
          {1, },
          c10::kDouble,
          c10::kStrided,
          c10::kCUDA,
          false);

  // Get CodeGen Runtime Arguments
  std::vector<CodeGen::CallArg> runArgs;
  runArgs.reserve(inputs.size() + 2);
  for (auto& input : inputs) {
    if (input.isDouble()) {
      runArgs.emplace_back(input.toDouble());
    } else if (input.isInt()) {
      runArgs.emplace_back(input.toInt());
    } else {
      runArgs.emplace_back(input.toTensor().data_ptr());
    }
  }
  runArgs.emplace_back(e_0_runtime.data_ptr());
  runArgs.emplace_back(e_1_runtime.data_ptr());
  runArgs.emplace_back(e_2_runtime.data_ptr());

  // CUDA KERNEL LAUNCH
  codegen_->call(runArgs);

  // Print Output
  std::cout << e_0_runtime << std::endl;
  std::cout << e_1_runtime << std::endl;
  std::cout << e_2_runtime << std::endl;
  std::cout << "Done!" << std::endl;

  return 0;
}


