/// Implements utilities for working with presburger analyses.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#include "dlnn-mlir/Dialect/DLNN/Utils/Presburger.h"

#include "dlnn-mlir/Dialect/DLNN/Utils/CommaSeparated.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::dlnn;
using namespace mlir::presburger;

[[nodiscard]] static uint64_t abs(int64_t x)
{
    return static_cast<uint64_t>(std::abs(x));
}

[[nodiscard]] static auto findSingleCoeff(ArrayRef<int64_t> row)
{
    return find_single_if(
        row,
        [](auto x) { return x != 0; });
}

int64_t asPositiveLhs(
    ArrayRef<int64_t> coeffs,
    unsigned idx,
    MutableArrayRef<int64_t> rhs)
{
    assert(idx >= 0 && idx < coeffs.size());

    // Obtain the value (in case rhs aliases coeffs).
    const auto val = coeffs[idx];

    // Obtain the signum of that coefficient, so we can ensure that it will
    // appear positive on the lhs.
    const auto sign = signum(val);
    const auto adjust = [&](int64_t x) { return -sign * x; };

    // The coefficients are moved to the right hand side.
    copy(map_range(coeffs, adjust), rhs.begin());
    // Remove the dimension that was involved.
    rhs[idx] = 0;

    // Return the original value (lhs is positive though).
    return val;
}

[[nodiscard]] std::pair<unsigned, int64_t> findSingleRelation(
    ArrayRef<int64_t> coeffs,
    unsigned offset,
    unsigned count,
    MutableArrayRef<int64_t> rhs)
{
    assert(offset >= 0 && offset < coeffs.size());
    assert(offset + count <= coeffs.size());
    assert(rhs.size() == coeffs.size());

    // Find the only non-zero coefficient in the subrange.
    const auto subrange = coeffs.slice(offset, count);
    const auto [it, single] = findSingleCoeff(subrange);

    if (it == subrange.end() || !single) return {no_index, {}}; // None found.

    // We found a match at the following index after offset.
    const auto idx = std::distance(subrange.begin(), it);
    const auto val = asPositiveLhs(coeffs, offset + idx, rhs);

    return {idx, val};
}

//===----------------------------------------------------------------------===//
// Space
//===----------------------------------------------------------------------===//

void Space::print(raw_ostream &os, bool asMap) const
{
    const auto [numDomain, numSymbols, numRange, numLocals] = std::tuple(
        getNumDomainIds(),
        getNumSymbolIds(),
        getNumRangeIds(),
        getNumLocalIds());

    const auto printDims = [&](unsigned count, const char* prefix) {
        os << CommaSeparated(map_range(
            iota_range<unsigned>(0, count, false),
            [&](auto dim) { return Twine(prefix).concat(Twine(dim)); }));
    };

    if (numDomain != 0 || asMap) {
        os << '(';
        printDims(numDomain, "i");
        os << ')';
    } else {
        os << '(';
        printDims(numRange, "d");
        os << ')';
    }
    if (numSymbols != 0) {
        os << '[';
        printDims(numSymbols, "s");
        os << ']';
    }
    if (numDomain != 0 || asMap) {
        os << " -> (";
        printDims(numRange, "o");
        os << ')';
    }
    if (numLocals != 0) {
        os << " E ";
        printDims(numLocals, "l");
    }
}

void Space::dump() const { errs() << *this << "\n"; }

//===----------------------------------------------------------------------===//
// AffineExprRef
//===----------------------------------------------------------------------===//

void AffineExprRef::print(raw_ostream &os) const
{
    const auto nonzero = make_filter_range(
        enumerate(getCoeffs()),
        [](auto &pair) { return pair.value() != 0; });
    const auto firstSymbol = getSpace().getIdKindOffset(IdKind::Symbol);
    const auto firstLocal = getSpace().getIdKindOffset(IdKind::Local);
    const auto constant = getCoeffs().size() - 1;
    const auto dimId = [&](unsigned offset) {
        if (offset == constant) return Twine("");
        if (offset >= firstLocal)
            return Twine("l").concat(Twine(offset - firstLocal));
        if (offset >= firstSymbol)
            return Twine("s").concat(Twine(offset - firstSymbol));
        return Twine("d").concat(Twine(offset));
    };

    auto first = true;
    for (auto [idx, val] : nonzero) {
        if (first && val < 0) os << '-';
        if (!first) os << (val < 0 ? " - " : " + ");
        first = false;
        const auto c = abs(val);
        if (idx == constant || c != 1) os << c;
        os << dimId(idx);
    }
    if (first) os << '0';
}

void AffineExprRef::dump() const { errs() << *this << "\n"; }

//===----------------------------------------------------------------------===//
// OptimisticBounds
//===----------------------------------------------------------------------===//

/// Combines the bound coefficients @p lhs and @p rhs .
template<bool Upper>
static int64_t combineBounds(int64_t lhs, int64_t rhs)
{
    if constexpr (Upper)
        return std::min(lhs, rhs);
    else
        return std::max(lhs, rhs);
}

/// Combines two bounds, establishing a tighter / better one.
template<bool Upper>
static void
naiveCombineBounds(MutableArrayRef<int64_t> existing, ArrayRef<int64_t> with)
{
    const auto next_coeff = [](auto first, auto last) {
        for (; first != last; ++first)
            if (*first != 0) return first;
        return last;
    };
    // If both are constant bounds, pick the tighter one.
    if (next_coeff(with.begin(), with.end()) == std::prev(with.end())) {
        if (next_coeff(existing.begin(), existing.end())
            == std::prev(existing.end()))
            existing.back() =
                combineBounds<Upper>(existing.back(), with.back());
        return;
    }

    // If one bound has less symbols / locals, use that one.
    const auto nonzero = [](int64_t x) { return x != 0; };
    if (count_if(existing.drop_back(1), nonzero)
        < count_if(existing.drop_back(1), nonzero))
        return;

    // Better one wins.
    copy(with, existing.begin());
}

LogicalResult
OptimisticBounds::combineLower(unsigned dim, ArrayRef<int64_t> bound)
{
    const auto maybeExisting = getLower(dim);
    if (!maybeExisting) {
        copy(bound, overwriteLower(dim).begin());
        return success();
    }

    // TODO: We need to do something better.
    naiveCombineBounds<false>(overwriteLower(dim), bound);

    return success();
}

LogicalResult
OptimisticBounds::combineUpper(unsigned dim, ArrayRef<int64_t> bound)
{
    const auto maybeExisting = getUpper(dim);
    if (!maybeExisting) {
        copy(bound, overwriteUpper(dim).begin());
        return success();
    }

    // TODO: We need to do something better.
    naiveCombineBounds<true>(overwriteLower(dim), bound);

    return success();
}

void OptimisticBounds::print(raw_ostream &os) const
{
    os << "OptimisticBounds(";
    getSpace().print(os);
    os << ") {\n";
    const auto rangeSpace = getRangeSpace();
    for (auto dim : iota_range<unsigned>(0, getRank(), false)) {
        const auto maybeLower = getLower(dim);
        const auto maybeUpper = getUpper(dim);
        if (!maybeLower && !maybeUpper) continue;

        if (maybeLower) os << AffineExprRef(rangeSpace, *maybeLower) << " <= ";
        os << "d" << dim;
        if (maybeUpper) os << " <= " << AffineExprRef(rangeSpace, *maybeUpper);
        os << "\n";
    }
    os << "}";
}

void OptimisticBounds::dump() const { errs() << *this << "\n"; }

//===----------------------------------------------------------------------===//
// OffsetsStridesSizes
//===----------------------------------------------------------------------===//

bool OffsetsStridesSizes::isDefinite() const
{
    const auto [offset, count] = std::pair(
        getDomainSpace().getNumSymbolIds(),
        getDomainSpace().getNumLocalIds());

    // No locals may appear in any of the rows.
    for (auto idx : iota_range<unsigned>(0, m_coeffs.getNumRows(), false))
        if (llvm::count(m_coeffs.getRow(idx).slice(offset, count), 0) != count)
            return false;

    return true;
}

void OffsetsStridesSizes::print(raw_ostream &os) const
{
    os << "OffsetsStridesSizes(";
    getDomainSpace().print(os);
    os << ") {\n";
    const auto rangeSpace = getRangeSpace();
    for (auto dim : iota_range<unsigned>(0, getRank(), false)) {
        os << "  d" << dim << ": +["
           << AffineExprRef(rangeSpace, getOffset(dim)) << "] *["
           << AffineExprRef(rangeSpace, getStride(dim)) << "] #["
           << AffineExprRef(rangeSpace, getSize(dim)) << "]\n";
    }
    os << "}";
}

void OffsetsStridesSizes::dump() const { errs() << *this << "\n"; }

//===----------------------------------------------------------------------===//
// Set
//===----------------------------------------------------------------------===//

std::optional<OptimisticBounds> Set::peelBounds()
{
    removeTrivialRedundancy();
    removeRedundantLocalVars();
    removeRedundantInequalities();

    // Initialize the result to the biggest representable bounds.
    OptimisticBounds result(getSpace());
    const auto [numDims, numSyms, numLocs] =
        std::tuple(getNumDimIds(), getNumSymbolIds(), getNumLocalIds());

    // Iterate over all inequalities.
    SmallVector<int64_t> bound(getNumCols());
    for (auto [first, last] = std::pair(0U, getNumInequalities());
         first != last;) {
        // Find the only dimension affected by this inequality.
        const auto ineq = getInequality(first++);
        const auto [dim, val] = findSingleRelation(ineq, 0, numDims, bound);
        if (dim == no_index) continue;

        if (failed(result.combine(
                val < 0 ? BoundType::UB : BoundType::LB,
                dim,
                ArrayRef<int64_t>(bound).drop_front(numDims))))
            return std::nullopt;

        // Removes the current inequality from the set.
        removeInequality(--first);
        --last;
    }

    return result;
}

std::optional<OffsetsStridesSizes> Set::peelOffsetsStridesSizes()
{
    removeTrivialRedundancy();
    removeRedundantLocalVars();

    OffsetsStridesSizes result(getSpace());
    const auto [numDims, numSyms, numLocs] =
        std::tuple(getNumDimIds(), getNumSymbolIds(), getNumLocalIds());

    // Assume a stride of 1 everywhere.
    for (auto dim : iota_range<unsigned>(0, numDims, false))
        result.getStride(dim).back() = 1;

    // Iterate over all equalities.
    SmallVector<int64_t> bound(getNumCols());
    for (auto [first, last] = std::pair(0U, getNumEqualities());
         first != last;) {
        // Find the only dimension affected by this equality.
        const auto eq = getEquality(first++);
        const auto [dim, val] = findSingleRelation(eq, 0, numDims, bound);
        if (dim == no_index) continue;

        // The offset is copied from these columns.
        copy(bound, result.getOffset(dim).begin());

        // Find the single local variable that it is bound to.
        const auto locs = eq.drop_front(numDims + numSyms);
        const auto [locIt, onlyLoc] = findSingleCoeff(locs);
        if (locIt == locs.end()) {
            // Since this means the RHS is constant, the stride is 0.
            result.getStride(dim).back() = 0;
            // Convert this equality into two inequalities so it can be picked
            // up by the bound calculation.
            addBound(Set::BoundType::LB, dim, eq.back());
            addBound(Set::BoundType::UB, dim, eq.back());
        } else if (!onlyLoc)
            return std::nullopt; // More than one!
        else {
            // The stride is the coefficient on this local.
            const auto locIdx = std::distance(locs.begin(), locIt);
            result.getStride(dim)[numDims + numSyms + locIdx] = *locIt;
            // But the local does not participate in the offset.
            result.getOffset(dim)[numDims + numSyms + locIdx] = 0;
        }

        // Removes the current inequality from the set.
        removeEquality(--first);
        --last;
    }

    return result;
}

std::optional<OffsetsStridesSizes> Set::toOffsetsStridesSizes() const
{
    Set peeled(*this);

    // Peel off equalities on the current dimensions and remove them.
    auto maybeResult = peeled.peelOffsetsStridesSizes();
    if (!maybeResult) return std::nullopt;

    // Peel off the bounds for the current dimensions.
    auto maybeBounds = peeled.peelBounds();
    if (!maybeBounds) return std::nullopt;

    if (!maybeResult->isDefinite()) {
        // TODO: Make a copy.
        // TODO: Delete the domain vars.
        // TODO: Promote the locals involved in maybeResult to dims.
        // TODO: Solve this problem for that subset.
        // TODO: Propagate the results upward.
        return std::nullopt;
    }

    if (!maybeBounds->isBounded()) return std::nullopt;

    // Calculate the effective bounds.
    for (auto dim : iota_range<unsigned>(0, getNumDimIds(), false)) {
        // The offset comes from the lower bound.
        if (const auto maybeLower = maybeBounds->getLower(dim)) {
            copy(*maybeLower, maybeResult->getOffset(dim).begin());

            // The size comes from the delta of lower and upper bound.
            if (const auto maybeUpper = maybeBounds->getUpper(dim)) {
                auto delta = to_vector(map_range(
                    iota_range<unsigned>(0, maybeLower->size(), false),
                    [&](auto i) {
                        return (*maybeUpper)[i] - (*maybeLower)[i];
                    }));
                delta.back() += 1;
                copy(delta, maybeResult->getSize(dim).begin());
            }
        }
    }

    return maybeResult;
}

void Set::print(raw_ostream &os) const
{
    getSpace().print(os);

    os << " : ";

    const auto numDims = getNumDimIds();
    SmallVector<int64_t> coeffs(getNumCols(), 0);

    interleaveComma(
        iota_range<unsigned>(0, getNumEqualities(), false),
        os,
        [&](auto eqIdx) {
            const auto eq = getEquality(eqIdx);
            const auto [idx, val] = findSingleRelation(eq, 0, numDims, coeffs);
            if (idx == no_index) {
                os << "0 = " << AffineExprRef(getSpace(), eq);
                return;
            }

            if (abs(val) != 1) os << abs(val);
            os << "d" << idx << " = " << AffineExprRef(getSpace(), coeffs);
        });
    if (getNumEqualities() > 0) os << ", ";
    interleaveComma(
        iota_range<unsigned>(0, getNumInequalities(), false),
        os,
        [&](auto ineqIdx) {
            const auto ineq = getInequality(ineqIdx);
            const auto [idx, val] =
                findSingleRelation(ineq, 0, numDims, coeffs);
            if (idx == no_index) {
                os << "0 <= " << AffineExprRef(getSpace(), ineq);
                return;
            }

            if (abs(val) != 1) os << abs(val);
            os << "d" << idx << (val < 0 ? " <= " : " >= ")
               << AffineExprRef(getSpace(), coeffs);
        });
}

void Set::dump() const { errs() << *this << "\n"; }

//===----------------------------------------------------------------------===//
// Map
//===----------------------------------------------------------------------===//

std::optional<Map> Map::fromAffineMap(AffineMap map)
{
    assert(map);

    // Use the MLIR built-in translation.
    FlatAffineRelation far;
    if (failed(getRelationFromMap(map, far))) return std::nullopt;

    // Restore the space structure to the result map, which MLIR drops.
    Map result(std::move(far));
    const auto space = Space::getRelationSpace(
        far.getNumDomainDims(),
        far.getNumRangeDims(),
        far.getNumSymbolIds(),
        far.getNumLocalIds());
    result.setSpace(space);
    return result;
}

Map Map::forTiling(ArrayRef<int64_t> tileSizes)
{
    // Construct the space for the tiling map.
    const auto numTileIndices = tileSizes.size() - count(tileSizes, 0);
    const auto rank = tileSizes.size();
    const auto space = Space::getRelationSpace(rank, rank, numTileIndices);

    // Create the tiling map.
    Map result(numTileIndices, rank, space.getNumIds() + 1, space);

    // Initialize the tiling map.
    auto tileDim = result.getIdKindOffset(IdKind::Symbol);
    SmallVector<int64_t> coeff(result.getNumCols(), 0);
    for (auto [dim, sz] : enumerate(tileSizes)) {
        // Add the tiling equality (which is identity if not tiled).
        coeff[dim] = -1;
        coeff[rank + dim] = 1;
        coeff[tileDim] = -sz;
        result.addEquality(coeff);
        coeff[rank + dim] = coeff[tileDim] = 0;

        if (sz != 0) {
            // Add the upper bound for the remainder.
            ++tileDim;
            coeff.back() = sz - 1;
            result.addInequality(coeff);
            coeff.back() = 0;
        }

        coeff[dim] = 0;
    }

    return result;
}

unsigned Map::findDirectEquality(unsigned dim, bool inverse) const
{
    const auto numEqs = getNumEqualities();
    const auto rangeOffset = getIdKindOffset(IdKind::Range);
    const auto offset = inverse ? rangeOffset : 0U;
    for (auto eqIdx : iota_range<unsigned>(0, numEqs, false))
        if (auto lhs = signum(atEq(eqIdx, offset + dim))) {
            const auto eq = inverse
                                ? getEquality(eqIdx).slice(0, rangeOffset)
                                : getEquality(eqIdx).drop_front(rangeOffset);
            if (any_of(eq, [lhs](auto c) { return signum(c) == -lhs; }))
                return eqIdx;
        }

    return no_index;
}

unsigned Map::promoteIndependentRangeIds()
{
    // Find all range variables that are not directly equality constrained
    // to some domain variables.
    const auto numDims = getNumDomainIds();
    auto unconstrained = to_vector(iota_range<unsigned>(0, numDims, false));
    for (auto j : iota_range<unsigned>(0, getNumEqualities(), false)) {
        const auto eq = getEquality(j);
        erase_if(unconstrained, [&](auto dim) {
            if (const auto lhs = signum(eq[dim]))
                return any_of(eq.drop_front(numDims), [lhs](auto c) {
                    return signum(c) == -lhs;
                });
            return false;
        });
    }

    /// Introduce new domain variables that map to the unconstrained range
    /// variables via an identity equality constraint.
    auto offset = appendId(IdKind::Range, unconstrained.size());
    SmallVector<int64_t> coeff(getNumCols(), 0);
    for (auto dim : unconstrained) {
        coeff[dim] = 1;
        coeff[offset] = -1;
        addEquality(coeff);
        coeff[dim] = coeff[offset++] = 0;
    }

    return unconstrained.size();
};

void Map::print(raw_ostream &os) const
{
    Space::getSetSpace(getNumDomainIds(), getNumSymbolIds(), getNumLocalIds())
        .print(os);

    // TODO: Implement.
    os << " -> NOT IMPLEMENTED, FALLBACK:\n";
    IntegerRelation::print(os);
}

void Map::dump() const { errs() << *this << "\n"; }

// namespace {

// using Stride = std::pair<unsigned, int64_t>;

// llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Stride
// &stride)
// {
//     if (stride.first == no_index) return os << stride.second;
//     return os << stride.second << "*l" << stride.first;
// }

// class OffsetsAndStrides {
// public:
//     explicit OffsetsAndStrides(const Space &domainSpace, unsigned rank)
//             : m_domainSpace(domainSpace),
//               m_offsets(rank, domainSpace.getNumIds() + 1),
//               m_strides(rank, {no_index, 1})
//     {
//         assert(domainSpace.getNumDomainIds() == 0 && "requires set
//         space");
//     }
//     explicit OffsetsAndStrides(const Set &set)
//             : OffsetsAndStrides(
//                 Space::getSetSpace(0, set.getNumSymbolIds()),
//                 set.getNumDimIds())
//     {}

//     [[nodiscard]] const Space &getDomainSpace() const { return
//     m_domainSpace;
//     }
//     [[nodiscard]] unsigned getRank() const { return
//     m_offsets.getNumRows(); }

//     [[nodiscard]] MutableArrayRef<int64_t> getOffset(unsigned dim)
//     {
//         return m_offsets.getRow(dim);
//     }
//     [[nodiscard]] ArrayRef<int64_t> getOffset(unsigned dim) const
//     {
//         return m_offsets.getRow(dim);
//     }

//     [[nodiscard]] const Stride &getStride(unsigned dim) const
//     {
//         return m_strides[dim];
//     }
//     [[nodiscard]] Stride &getStride(unsigned dim) { return
//     m_strides[dim]; }

//     void dump() const
//     {
//         getDomainSpace().dump();
//         const auto exprSpace =
//             Space::getSetSpace(0, m_offsets.getNumColumns() - 1);
//         for (auto dim : iota_range<unsigned>(0, getRank(), false)) {
//             errs() << "d" << dim << ":\n";
//             errs() << "  " << AffineExprRef(exprSpace, getOffset(dim)) <<
//             "\n"; errs() << "  " << getStride(dim) << "\n";
//         }
//     }

// private:
//     Space m_domainSpace;
//     Matrix m_offsets;
//     SmallVector<Stride> m_strides;
// };

// } // namespace

// static std::optional<OffsetsAndStrides> peelOffsetsAndStrides(Set &set)
// {
//     // Initialize the result to offset 0 and stride 1, and no dims bound.
//     OffsetsAndStrides result(set);
//     const auto [numDims, numSyms, numLocs] = std::tuple(
//         set.getNumDimIds(),
//         set.getNumSymbolIds(),
//         set.getNumLocalIds());
//     SmallBitVector boundDims(numDims, false);

//     // Iterate over all equalities in set.
//     for (auto [first, last] = std::pair(0U, set.getNumEqualities());
//          first != last;) {
//         // Get and split the equality into parts.
//         const auto eq = set.getEquality(first++);
//         const auto [dims, syms, locs] = std::tuple(
//             eq.slice(0, numDims),
//             eq.slice(numDims, numSyms),
//             eq.slice(numDims + numSyms, numLocs));

//         // Find the single, unbound dimension that this equality
//         constrains. const auto [dimIt, onlyDim] = findSingleCoeff(dims);
//         if (dimIt == dims.end()) continue; // None found, ignore.
//         if (!onlyDim) return std::nullopt; // More than one!
//         const auto dimIdx = std::distance(eq.begin(), dimIt);
//         if (boundDims.test(dimIdx)) return std::nullopt; // Already
//         bound! boundDims.set(dimIdx);

//         // All other coefficients need to adjust their signs accordingly.
//         const auto sign = signum(*dimIt);
//         const auto adjust_sign = [&](auto x) { return -sign * x; };

//         // Find the single local variable that it is bound to.
//         const auto [locIt, onlyLoc] = findSingleCoeff(locs);
//         if (locIt == locs.end()) {
//             // Since this means the RHS is constant, the stride is 0.
//             result.getStride(dimIdx) = {no_index, 0};
//             // Convert this equality into two inequalities so it can be
//             picked
//             // up by the bound calculation.
//             set.addBound(Set::BoundType::LB, dimIdx, eq.back());
//             set.addBound(Set::BoundType::UB, dimIdx, eq.back());
//         } else if (!onlyLoc)
//             return std::nullopt; // More than one!
//         else {
//             // The stride is the coefficient on this local with opposing
//             sign. const auto locIdx = std::distance(locs.begin(), locIt);
//             result.getStride(dimIdx) = {locIdx, adjust_sign(*locIt)};
//         }

//         // The offset is copied from the remaining columns.
//         copy(map_range(syms, adjust_sign),
//         result.getOffset(dimIdx).begin());
//         result.getOffset(dimIdx).back() = adjust_sign(eq.back());

//         // Removes the current equality from the set.
//         set.removeEquality(--first);
//         --last;
//     }

//     return result;
// }

// std::optional<OffsetsStridesSizes> Set::toOffsetsStridesSizes() const
// {
//     OffsetsStridesSizes result(
//         Space::getSetSpace(getNumDimIds(), getNumSymbolIds()));

//     const auto rank = getNumDimIds();
//     Set peeled(*this);

//     // Peel off equalities on the current dimensions and remove them.
//     peeled.removeTrivialRedundancy();
//     peeled.removeRedundantLocalVars();
//     auto maybeOffsetsAndStrides = peelOffsetsAndStrides(peeled);
//     if (!maybeOffsetsAndStrides) return std::nullopt;

//     // Peel off the bounds for the current dimensions.
//     peeled.removeRedundantInequalities();
//     auto maybeBounds = peeled.peelBounds();
//     if (!maybeBounds) return std::nullopt;
//     maybeBounds->dump();

//     // Initialize the result under the assumption that no equalities
//     apply. for (auto dim : iota_range<unsigned>(0, rank, false)) {
//         // Strides are known to be 1 here.
//         result.getStride(dim).back() = 1;

//         // The offset comes from the lower bound.
//         if (const auto maybeLower = maybeBounds->getLower(dim)) {
//             copy(*maybeLower, result.getOffset(dim).begin());

//             // The size comes from the delta of lower and upper bound.
//             if (const auto maybeUpper = maybeBounds->getUpper(dim)) {
//                 auto delta = to_vector(map_range(
//                     iota_range<unsigned>(0, maybeLower->size(), false),
//                     [&](auto i) {
//                         return (*maybeUpper)[i] - (*maybeLower)[i];
//                     }));
//                 delta.back() += 1;
//                 copy(delta, result.getSize(dim).begin());
//             }
//         }
//     }

//     if (getNumEqualities() != 0) {
//         return std::nullopt;
//         // peeled.removeIdRange(IdKind::Domain, 0, rank);

//         // // Formulate a subproblem on the locals that were used.
//         // unsigned newDim = 0;
//         // DenseMap<unsigned, unsigned> localMap;
//         // for (auto dim : iota_range<unsigned>(0, rank, false)) {
//         //     const auto [locIdx, _] =
//         maybeOffsetsAndStrides->getStride(dim);
//         //     if (locIdx == no_index) continue;
//         //     peeled.convertIdKind(IdKind::Local, locIdx, locIdx + 1,
//         //     IdKind::Domain); localMap[locIdx] = newDim++;
//         // }

//         // // Solve the subproblem.
//         // auto maybeLocalOffsetsStridesSizes =
//         peeled.toOffsetsStridesSizes();
//         // if (!maybeLocalOffsetsStridesSizes) return std::nullopt;

//         // // Transfer the results.
//         // for (auto dim : iota_range<unsigned>(0, rank, false)) {
//         //     const auto [locIdx, stride] =
//         //     maybeOffsetsAndStrides->getStride(dim); if (locIdx ==
//         no_index) {
//         //         // The original assumption held.
//         //         continue;
//         //     }
//         //     const auto newIdx = localMap[locIdx];
//         //     const auto adjust = [stride = stride](int64_t x) { return
//         x *
//         //     stride;
//         //     };

//         //     // Inherit stride and size from the local solution,
//         adjusted for
//         //     the
//         //     // discovered stride.
//         //     const auto locStride =
//         //     maybeLocalOffsetsStridesSizes->getStride(newIdx);
//         //     copy(map_range(locStride, adjust),
//         //     result.getStride(dim).begin()); const auto locSize =
//         //     maybeLocalOffsetsStridesSizes->getSize(newIdx);
//         //     copy(map_range(locSize, adjust),
//         result.getSize(dim).begin());

//         //     // const auto locOffset =
//         //     // maybeLocalOffsetsStridesSizes->getOffset(newIdx);
//         //     // copy(map_range(locOffset, adjust),
//         //     result.getOffset(dim).begin());
//         // }
//     }

//     if (!maybeBounds->isBounded()) return std::nullopt;

//     return result;
// }

void SetAndSymbolValues::dump() const
{
    errs() << "Symbols: [";
    interleaveComma(m_symbolValues, errs());
    errs() << "]\n";
    getSet().dump();
}
