#include "ir_utils.h"

namespace torch {
namespace jit {
namespace tensorexpr {

namespace {

class ReduceChecker : public IRVisitor {
 public:
  ReduceChecker(const VarPtr &iv) : iv(iv) {}

  void visit(ReduceOpPtr reduce) {
    auto &reduceArgs = reduce->reduce_args();
    isReduce |= std::count(reduceArgs.begin(), reduceArgs.end(), iv);
  }

  operator bool() const { return isReduce; }

 private:
  VarPtr iv;
  bool isReduce = false;
};

}  // namespace

bool isReductionLoop(ForPtr loop) {
  ReduceChecker checker(loop->var());
  loop->body()->accept(&checker);
  return checker;
}

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch