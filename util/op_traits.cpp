#include "op_traits.h"

namespace torch {
namespace jit {

std::unordered_set<Symbol> supportedViewOpSymbols{
    // Selection
    aten::select,
    // Slice
    aten::slice,
    // Reshape
    aten::squeeze, aten::unsqueeze, aten::reshape, aten::view,
    // Expansion
    aten::expand,
    // Permutation
    aten::permute,
    // Advanced Indexing (not really a view, but regarded as a sparse view in
    // our work)
    aten::index};

}
}  // namespace torch