/// Declares additional template library utilities.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#pragma once

#include "llvm/ADT/STLExtras.h"

#include <concepts>
// BUG: clang++-12 can't deal with libstdc++'s implementation.
// #include <ranges>
#include <tuple>
#include <type_traits>

namespace mlir::dlnn {

//===----------------------------------------------------------------------===//
// Check that [first, last) is an iota range.
//===----------------------------------------------------------------------===//

// template<std::forward_iterator It, std::sentinel_for<It> S, class T>
template<class It, class S, class T>
inline constexpr bool is_iota(It first, S last, T init)
{
    for (; first != last; ++first)
        if (static_cast<T>(*first) != init++) return false;
    return true;
}

// template<std::ranges::range R, class T>
template<class R, class T>
inline constexpr bool is_iota(R &&range, T &&init)
{
    return is_iota(
        // std::ranges::begin(range),
        range.begin(),
        // std::ranges::end(range),
        range.end(),
        std::forward<T>(init));
}

} // namespace mlir::dlnn

//===----------------------------------------------------------------------===//
// Finds the only element matching pred in [first, last).
//===----------------------------------------------------------------------===//

template<std::forward_iterator It, std::sentinel_for<It> S, class Pred>
inline constexpr std::pair<It, bool> find_single_if(It first, S last, Pred pred)
{
    const auto it = std::find_if(first, last, pred);
    if (it == last) return {last, true};
    if (std::find_if(std::next(it), last, pred) != last) return {it, false};
    return {it, true};
}

// template<std::ranges::range R, class Pred>
template<class R, class Pred>
inline constexpr auto find_single_if(R &&range, Pred pred)
    // -> decltype(find_single_if(std::ranges::begin(range),
    // std::ranges::end(range), pred))
    -> decltype(find_single_if(range.begin(), range.end(), pred))
{
    return find_single_if(
        // std::ranges::begin(range),
        range.begin(),
        // std::ranges::end(range),
        range.end(),
        pred);
}

//===----------------------------------------------------------------------===//
// Enable structural binding for llvm::enumerate()
//===----------------------------------------------------------------------===//

template<class R>
struct std::tuple_size<llvm::detail::result_pair<R>>
        : std::integral_constant<std::size_t, 2> {};
template<class R>
struct std::tuple_element<0, llvm::detail::result_pair<R>> {
    using type = std::size_t;
};
template<class R>
struct std::tuple_element<1, llvm::detail::result_pair<R>> {
    using type = typename llvm::detail::result_pair<R>::value_reference;
};

namespace llvm::detail {

template<std::size_t N, class R>
requires(N < 2)
    [[nodiscard]] inline constexpr auto get(const result_pair<R> &pair)
{
    if constexpr (N == 0)
        return pair.index();
    else
        return pair.value();
}

} // namespace llvm::detail

namespace mlir::dlnn {

//==-----------------------------------------------------------------------===//
// signum()
//==-----------------------------------------------------------------------===//

/// Obtains the signum {-1, 0, 1} of a signed @p value .
template<std::signed_integral SInt>
[[nodiscard]] constexpr SInt signum(SInt value)
{
    return static_cast<SInt>((value > 0) - (value < 0));
}
/// Obtains the signum {0, 1} of an unsigned @p value .
template<std::unsigned_integral UInt>
[[nodiscard]] constexpr UInt signum(UInt value)
{
    return static_cast<UInt>(value > 0);
}

} // namespace mlir::dlnn
