#include "collect_interm_buf.h"

namespace torch {
namespace jit {
namespace tensorexpr {

class IntermBufCollector : public IRVisitor {
 public:
  IntermBufCollector(const std::unordered_set<BufPtr> &outBufs)
      : outBufs(outBufs) {}

  auto &&getIntermBufs() && { return std::move(intermBufs); }

  void visit(StorePtr store) override {
    auto buf = store->buf();
    if (outBufs.count(buf)) return;
    if (std::count(intermBufs.begin(), intermBufs.end(), buf)) return;
    intermBufs.push_back(buf);
  }

 private:
  const std::unordered_set<BufPtr> &outBufs;
  std::vector<BufPtr> intermBufs;
};

std::vector<BufPtr> collectIntermBufs(
    const StmtPtr &stmt, const std::unordered_set<BufPtr> &outBufs) {
  IntermBufCollector collector(outBufs);
  stmt->accept(&collector);
  return std::move(collector).getIntermBufs();
}

}  // namespace tensorexpr
}  // namespace jit
}  // namespace torch