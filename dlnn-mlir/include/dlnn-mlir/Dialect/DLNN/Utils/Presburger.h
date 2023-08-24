/// Declares utilities for working presburger analyses.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#pragma once

#include "dlnn-mlir/Dialect/DLNN/Utils/STLExtras.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"

#include "llvm/ADT/SmallBitVector.h"

#include <limits>
#include <optional>

namespace mlir::dlnn {

/// Constant that represents no index found.
inline constexpr unsigned no_index = std::numeric_limits<unsigned>::max();

using mlir::presburger::IdKind;
using mlir::presburger::IntegerPolyhedron;
using mlir::presburger::IntegerRelation;
using mlir::presburger::PresburgerSpace;

/// Derives from PresburgerSpace to provide some debugging functionality.
class [[nodiscard]] Space : public PresburgerSpace {
public:
    /// Obtains a Space for a Set.
    static Space getSetSpace(
        unsigned numDims = 0,
        unsigned numSymbols = 0,
        unsigned numLocals = 0)
    {
        return Space(0, numDims, numSymbols, numLocals);
    }
    /// Obtains a Space for a Map.
    static Space getRelationSpace(
        unsigned numDomain = 0,
        unsigned numRange = 0,
        unsigned numSymbols = 0,
        unsigned numLocals = 0)
    {
        return Space(numDomain, numRange, numSymbols, numLocals);
    }

    /*implicit*/ Space(const PresburgerSpace &space) : PresburgerSpace(space) {}

    /// Determines whether this Space can be used for a Set.
    [[nodiscard]] bool isSet() const { return getNumDomainIds() == 0; }

    /// Prints a human-readable representation to @p os .
    void print(llvm::raw_ostream &os, bool asMap = false) const;
    /// Prints this Space to STDERR.
    void dump() const;

    /// @copydoc print()
    friend llvm::raw_ostream &
    operator<<(llvm::raw_ostream &os, const Space &space)
    {
        space.print(os);
        return os;
    }

private:
    using PresburgerSpace::PresburgerSpace;
};

/// Wraps an ArrayRef on coefficients together with their space.
///
/// @warning    Neither the space nor the coefficients are owned!
class [[nodiscard]] AffineExprRef {
public:
    /// Initializes an AffineExprRef for @p space and @p coeffs .
    ///
    /// @pre        `space.isSet()`
    /// @pre        `coeffs.size() == space.getNumIds() + 1`
    explicit AffineExprRef(const Space &space, ArrayRef<int64_t> coeffs)
            : m_space(space),
              m_coeffs(coeffs)
    {
        assert(space.isSet() && "requires set space");
        assert(coeffs.size() == space.getNumIds() + 1 && "invalid coeffs");
    }

    /// Gets the associated space.
    [[nodiscard]] const Space &getSpace() const { return m_space; }
    /// Gets the coefficients.
    [[nodiscard]] ArrayRef<int64_t> getCoeffs() const { return m_coeffs; }

    /// Prints a human-readable representation to @p os .
    void print(llvm::raw_ostream &os) const;
    /// Prints this AffineExprRef to STDERR.
    void dump() const;

    /// @copydoc print()
    friend llvm::raw_ostream &
    operator<<(llvm::raw_ostream &os, const AffineExprRef &ref)
    {
        ref.print(os);
        return os;
    }

private:
    const Space &m_space;
    ArrayRef<int64_t> m_coeffs;
};

/// Stores a list of affine expressions that track the bounds of set dimensions.
class [[nodiscard]] OptimisticBounds {
public:
    using BoundType = IntegerRelation::BoundType;

    /// Initializes OptimisticBounds that are unbounded.
    explicit OptimisticBounds(const Space &space)
            : m_space(space),
              m_expressions(
                  2 * space.getNumDimIds(),
                  space.getNumSymbolIds() + space.getNumLocalIds() + 1),
              m_bounded(2 * space.getNumDimIds(), false)
    {
        assert(space.getNumDomainIds() == 0 && "requires set space");
    }

    /// Gets the associated set space.
    [[nodiscard]] const Space &getSpace() const { return m_space; }
    /// Gets the associated range space.
    [[nodiscard]] Space getRangeSpace() const
    {
        return Space::getSetSpace(
            0,
            getSpace().getNumSymbolIds(),
            getSpace().getNumLocalIds());
    }

    /// Gets the number of set dimensions.
    [[nodiscard]] unsigned getRank() const { return getSpace().getNumDimIds(); }

    /// Determines whether all dimensions are bounded.
    [[nodiscard]] bool isBounded() const { return m_bounded.all(); }

    [[nodiscard]] bool hasLower(unsigned dim) const
    {
        assert(dim < getRank() && "dim out of range");
        return m_bounded.test(dim);
    }
    [[nodiscard]] std::optional<ArrayRef<int64_t>> getLower(unsigned dim) const
    {
        assert(dim < getRank() && "dim out of range");
        if (!hasLower(dim)) return std::nullopt;
        return m_expressions.getRow(dim);
    }
    [[nodiscard]] MutableArrayRef<int64_t> overwriteLower(unsigned dim)
    {
        assert(dim < getRank() && "dim out of range");
        m_bounded.set(dim);
        return m_expressions.getRow(dim);
    }
    LogicalResult combineLower(unsigned dim, ArrayRef<int64_t> bound);

    [[nodiscard]] bool hasUpper(unsigned dim) const
    {
        assert(dim < getRank() && "dim out of range");
        return m_bounded.test(getRank() + dim);
    }
    [[nodiscard]] std::optional<ArrayRef<int64_t>> getUpper(unsigned dim) const
    {
        assert(dim < getRank() && "dim out of range");
        if (!hasUpper(dim)) return std::nullopt;
        return m_expressions.getRow(getRank() + dim);
    }
    [[nodiscard]] MutableArrayRef<int64_t> overwriteUpper(unsigned dim)
    {
        assert(dim < getRank() && "dim out of range");
        m_bounded.set(getRank() + dim);
        return m_expressions.getRow(getRank() + dim);
    }
    LogicalResult combineUpper(unsigned dim, ArrayRef<int64_t> bound);

    LogicalResult combine(BoundType type, unsigned dim, ArrayRef<int64_t> bound)
    {
        const auto lower = type == BoundType::LB || type == BoundType::EQ
                               ? combineLower(dim, bound)
                               : success();
        const auto upper = type == BoundType::UB || type == BoundType::EQ
                               ? combineUpper(dim, bound)
                               : success();
        return success(succeeded(lower) && succeeded(upper));
    }

    /// Prints a human-readable representation to @p os .
    void print(llvm::raw_ostream &os) const;
    /// Prints this OptimisticBounds to STDERR.
    void dump() const;

    /// @copydoc print()
    friend llvm::raw_ostream &
    operator<<(llvm::raw_ostream &os, const OptimisticBounds &ob)
    {
        ob.print(os);
        return os;
    }

private:
    Space m_space;
    presburger::Matrix m_expressions;
    llvm::SmallBitVector m_bounded;
};

/// Stores affine expressions for offsets, strides and sizes of dimensions.
class [[nodiscard]] OffsetsStridesSizes {
public:
    /// Initializes undefined OffsetsStridesSizes.
    ///
    /// @pre        `domainSpace.isSet()`
    /// @pre        `domainSpace.getNumLocalIds() == 0`
    explicit OffsetsStridesSizes(const Space &domainSpace)
            : m_domainSpace(domainSpace),
              m_coeffs(
                  3 * domainSpace.getNumDimIds(),
                  domainSpace.getNumSymbolIds() + domainSpace.getNumLocalIds()
                      + 1)
    {
        assert(domainSpace.isSet() && "requires set space");
    }

    /// Gets the associated domain space.
    [[nodiscard]] const Space &getDomainSpace() const { return m_domainSpace; }
    /// Gets the number of set dimensions.
    [[nodiscard]] unsigned getRank() const
    {
        return getDomainSpace().getNumDimIds();
    }
    /// Gets the associated range space.
    [[nodiscard]] Space getRangeSpace() const
    {
        return Space::getSetSpace(
            0,
            getDomainSpace().getNumSymbolIds()
                + getDomainSpace().getNumLocalIds());
    }

    void insertSymbols(unsigned pos, unsigned count)
    {
        assert(pos <= getDomainSpace().getNumSymbolIds());
        m_domainSpace.insertId(IdKind::Symbol, pos, count);
        m_coeffs.insertColumns(pos, count);
    }
    void stripLocals()
    {
        const auto offset = m_domainSpace.getNumSymbolIds();
        const auto count = m_domainSpace.getNumLocalIds();
        m_domainSpace.removeIdRange(IdKind::Local, 0, count);
        m_coeffs.removeColumns(offset, count);
    }

    /// Determines whether no locals are involved in the defined expressions.
    [[nodiscard]] bool isDefinite() const;

    [[nodiscard]] MutableArrayRef<int64_t> getOffset(unsigned dim)
    {
        assert(dim < getRank() && "dim out of range");
        return m_coeffs.getRow(dim);
    }
    [[nodiscard]] ArrayRef<int64_t> getOffset(unsigned dim) const
    {
        assert(dim < getRank() && "dim out of range");
        return m_coeffs.getRow(dim);
    }
    [[nodiscard]] MutableArrayRef<int64_t> getStride(unsigned dim)
    {
        assert(dim < getRank() && "dim out of range");
        return m_coeffs.getRow(getRank() + dim);
    }
    [[nodiscard]] ArrayRef<int64_t> getStride(unsigned dim) const
    {
        assert(dim < getRank() && "dim out of range");
        return m_coeffs.getRow(getRank() + dim);
    }
    [[nodiscard]] MutableArrayRef<int64_t> getSize(unsigned dim)
    {
        assert(dim < getRank() && "dim out of range");
        return m_coeffs.getRow(2 * getRank() + dim);
    }
    [[nodiscard]] ArrayRef<int64_t> getSize(unsigned dim) const
    {
        assert(dim < getRank() && "dim out of range");
        return m_coeffs.getRow(2 * getRank() + dim);
    }

    /// Prints a human-readable representation to @p os .
    void print(llvm::raw_ostream &os) const;
    /// Prints this OffsetsStridesSizes to STDERR.
    void dump() const;

    /// @copydoc print()
    friend llvm::raw_ostream &
    operator<<(llvm::raw_ostream &os, const OffsetsStridesSizes &oss)
    {
        oss.print(os);
        return os;
    }

private:
    Space m_domainSpace;
    presburger::Matrix m_coeffs;
};

/// Derives from IntegerPolyhedron to provide some additional functionality.
class [[nodiscard]] Set : public mlir::presburger::IntegerPolyhedron {
public:
    using IntegerPolyhedron::IntegerPolyhedron;
    /*implicit*/ Set(IntegerPolyhedron &&move)
            : IntegerPolyhedron(std::move(move))
    {}

    void removeRedundantLocalVars()
    {
        IntegerPolyhedron::removeRedundantLocalVars();
    }

    /// Attempts to derive optimistic bounds for all dimensions in the set.
    ///
    /// Inequalities that were used in this process are removed from the set.
    std::optional<OptimisticBounds> peelBounds();
    /// Attempts to derive OffsetsStridesSizes for all dimensions in the set.
    ///
    /// Equalities that were used in this process are removed from the set, and
    /// new inequalities may be added.
    std::optional<OffsetsStridesSizes> peelOffsetsStridesSizes();

    /// Attempts to extract an OffsetsStridesSizes representation for this set.
    std::optional<OffsetsStridesSizes> toOffsetsStridesSizes() const;

    Space getSpace() const { return Space(IntegerPolyhedron::getSpace()); }

    /// Prints a human-readable representation to @p os .
    void print(llvm::raw_ostream &os) const;
    /// Prints this Set to STDERR.
    void dump() const;

    /// @copydoc print()
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Set &set)
    {
        set.print(os);
        return os;
    }
};

class [[nodiscard]] SetAndSymbolValues {
public:
    explicit SetAndSymbolValues(Set set)
            : m_set(std::move(set)),
              m_symbolValues()
    {
        m_symbolValues.resize(getSet().getNumSymbolIds(), Value{});
    }
    explicit SetAndSymbolValues(Set set, SmallVectorImpl<Value> &&symbolValues)
            : m_set(std::move(set)),
              m_symbolValues(std::move(symbolValues))
    {
        assert(m_symbolValues.size() == getSet().getNumSymbolIds());
    }

    const Set &getSet() const { return m_set; }
    /*implicit*/ operator const Set&() const { return getSet(); }

    ArrayRef<Value> getSymbolValues() const { return m_symbolValues; }
    void setSymbolValue(unsigned idx, Value value)
    {
        m_symbolValues[idx] = value;
    }
    Value getSymbolValue(unsigned idx) const { return m_symbolValues[idx]; }

    void dump() const;

private:
    Set m_set;
    SmallVector<Value> m_symbolValues;
};

/// Determines whether @p row contains a simple equality, i.e. -1 <-> 1 pair.
[[nodiscard]] bool isSimpleEquality(ArrayRef<int64_t> row);

/// Derives from IntegerRelation to provide some additional functionality.
class [[nodiscard]] Map : public mlir::presburger::IntegerRelation {
public:
    /// Tries to construct a Map from an AffineMap.
    [[nodiscard]] static std::optional<Map> fromAffineMap(AffineMap map);
    /// Constructs a tiling map for @p tileSizes .
    ///
    /// The resulting maps project from tiled space to original space by
    /// introducing a symbol for every tile index in the order they appear as
    /// non-zero entries in @p tileSizes . Additionally, inequalities for the
    /// remainder dimensions are inserted.
    ///
    /// `(d0, ..., dM)[t0, ..., tN] -> (d0, ..., si*ti + di, ..., dM) : di < si`
    static Map forTiling(ArrayRef<int64_t> tileSizes);

    using IntegerRelation::IntegerRelation;
    /*implicit*/ Map(IntegerRelation &&move) : IntegerRelation(std::move(move))
    {}

    /// Finds the first direct equality constraint for @p dim .
    ///
    /// If @p inverse is @c false , @p dim is a domain variable and the needle
    /// is a range variable, otherwise the vice versa.
    ///
    /// @retval     no_index    None found.
    /// @retval     unsigned    The index of the found equality constraint.
    [[nodiscard]] unsigned
    findDirectEquality(unsigned dim, bool inverse = false) const;

    void removeRedundantLocalVars()
    {
        IntegerRelation::removeRedundantLocalVars();
    }

    /// Introduces new domain variables that map to range variables that do not
    /// have any direct equality constraints with the domain placed on them.
    ///
    /// @return     Number of new domain variables.
    unsigned promoteIndependentRangeIds();

    Space getSpace() const { return Space(IntegerRelation::getSpace()); }

    /// Changes the space of this map.
    ///
    /// @pre        `space.getNumIds() == getNumIds()`
    void setSpace(const Space &space)
    {
        assert(space.getNumIds() == getNumIds());
        this->space = space;
    }

    /// Prints a human-readable representation to @p os .
    void print(llvm::raw_ostream &os) const;
    /// Prints this Map to STDERR.
    void dump() const;

    /// @copydoc print()
    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Map &map)
    {
        map.print(os);
        return os;
    }
};

} // namespace mlir::dlnn
