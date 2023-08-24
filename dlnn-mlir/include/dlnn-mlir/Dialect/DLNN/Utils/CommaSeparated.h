/// Declares a printable wrapper for calling interleaveComma().
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/IR/Diagnostics.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"

#include <type_traits>

namespace mlir::dlnn {

/// Wrapper around an iterator range to print it separated with commas.
template<class Range>
struct CommaSeparated {
    CommaSeparated(auto &&range) : range(std::forward<decltype(range)>(range))
    {}
    Range range;

    /// Writes the contents of @p range separated by commas to @p os .
    template<class Out>
    friend Out &operator<<(Out &os, const CommaSeparated<Range> &range)
    {
        interleaveComma(range.range, os);
        return os;
    }
};

template<class Container>
CommaSeparated(Container) -> CommaSeparated<std::remove_reference_t<Container>>;

} // namespace mlir::dlnn
