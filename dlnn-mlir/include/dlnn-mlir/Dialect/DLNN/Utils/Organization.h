/// Declares utilities for working with organized types.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#pragma once

#include "dlnn-mlir/Dialect/DLNN/IR/Base.h"
#include "dlnn-mlir/Dialect/DLNN/Utils/STLExtras.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

namespace mlir::dlnn {

/// Obtains the default organization for @p dims number of dims.
///
/// Constructs the map:
///
///     `(d0, ..., di)[N] -> (N, d0, ..., di)`
///
/// If @p batch is @c true , or without `N` if it is @c false .
inline AffineMap
getDefaultOrganization(MLIRContext* ctx, unsigned dims, bool batch)
{
    SmallVector<AffineExpr> results;
    if (batch) results.push_back(getAffineSymbolExpr(0, ctx));
    for (auto dim : llvm::iota_range<unsigned>(0, dims, false))
        results.push_back(getAffineDimExpr(dim, ctx));

    return AffineMap::get(
        dims,
        /*symbolCount=*/batch ? 1 : 0,
        results,
        ctx);
}

/// Determines whether @p map is a default organization map.
///
/// See getDefaultOrganization() for more details.
inline bool isDefaultOrganization(AffineMap map)
{
    // Expected number of symbols & dimensions.
    if (map.getNumSymbols() != 1) return false;
    if (map.getNumResults() != map.getNumInputs()) return false;

    // Expect leading batch dimension.
    auto batch = map.getResult(0).dyn_cast<AffineSymbolExpr>();
    if (!batch || batch.getPosition() != 0) return false;

    // Expect trailing identity.
    for (auto [dim, expr] : llvm::enumerate(map.getResults())) {
        auto dimExpr = expr.dyn_cast<AffineDimExpr>();
        if (!dimExpr || dimExpr.getPosition() != dim - 1) return false;
    }

    return true;
}

/// Describes the domain of a feature map.
///
/// A feature map is a structured type with a multi-dimensional shape
/// characterized by:
///     - Exactly one static, non-empty dimension with "channel" semantics `c`.
///     - An arbitrary amount of static or dynamic, non-empty dimensions with
///       "size" semantics `x_i`.
struct FeatureMapDomain {
public:
    static LogicalResult verify(
        function_ref<InFlightDiagnostic()> emitError,
        unsigned numChannels,
        ShapeRef sizes)
    {
        if (numChannels <= 0)
            return emitError() << "expected 1 or more channels";

        for (auto dim : llvm::iota_range<unsigned>(0, sizes.size(), false)) {
            if (sizes[dim] <= 0 && sizes[dim] != dynamic_size)
                return emitError() << "invalid size (" << sizes[dim]
                                   << ") for dimension #" << dim;
        }

        return success();
    }

    /// Initializes a FeatureMapDomain.
    ///
    /// @pre        `numChannels > 0`
    /// @pre        `sizes[i] > 0 || sizes[i] == dynamic_size`
    explicit FeatureMapDomain(unsigned numChannels, ShapeRef sizes)
            : m_numChannels(numChannels),
              m_sizes(sizes)
    {
        assert(numChannels > 0 && "empty channel dimension");
        assert(
            llvm::all_of(
                sizes,
                [](auto dim) { return dim > 0 || dim == dynamic_size; })
            && "invalid size dimension");
    }

    /// Gets the number of channels.
    unsigned getNumChannels() const { return m_numChannels; }
    /// Gets the size-like dimensions.
    ShapeRef getSizes() const { return m_sizes; }

    /// Obtains the default organization for this domain.
    AffineMap getDefaultOrganization(MLIRContext* ctx) const
    {
        return dlnn::getDefaultOrganization(ctx, 1 + getSizes().size(), true);
    }

    bool operator==(const FeatureMapDomain &) const = default;

    friend llvm::hash_code hash_value(const FeatureMapDomain &domain)
    {
        return llvm::hash_combine(domain.getNumChannels(), domain.getSizes());
    }

private:
    unsigned m_numChannels;
    ShapeRef m_sizes;
};

/// Describes the domain of a weights set.
///
/// A set of weights is a structured type with a multi-dimensional shape
/// characterized by:
///     - Exactly one static, non-empty dimension with "input channels"
///       semantics `c`.
///     - Exactly one static, non-empty dimension with "output channels"
///       semantics `f`.
///     - An arbitrary amount of static, non-empty dimensions with "size"
///       semantics `x_i`.
struct WeightsDomain {
    static LogicalResult verify(
        function_ref<InFlightDiagnostic()> emitError,
        unsigned numInChannels,
        unsigned numOutChannels,
        ShapeRef sizes)
    {
        if (numInChannels <= 0)
            return emitError() << "expected 1 or more input channels";
        if (numOutChannels <= 0)
            return emitError() << "expected 1 or more output channels";

        for (auto [dim, sz] : llvm::enumerate(sizes)) {
            if (sz <= 0)
                return emitError()
                       << "invalid size (" << sz << ") for dimension #" << dim;
        }

        return success();
    }

    /// Initializes a WeightsDomain.
    ///
    /// @pre        `numInChannels > 0`
    /// @pre        `numOutChannels > 0`
    /// @pre        `sizes[i] > 0`
    explicit WeightsDomain(
        unsigned numInChannels,
        unsigned numOutChannels,
        ShapeRef sizes)
            : m_numInChannels(numInChannels),
              m_numOutChannels(numOutChannels),
              m_sizes(sizes)
    {
        assert(numInChannels > 0 && "empty input channel dimension");
        assert(numOutChannels > 0 && "empty output channel dimension");
        assert(
            llvm::all_of(sizes, [](auto dim) { return dim > 0; })
            && "invalid size dimension");
    }

    /// Gets the number of input channels.
    unsigned getNumInChannels() const { return m_numInChannels; }
    /// Gets the number of output channels.
    unsigned getNumOutChannels() const { return m_numOutChannels; }
    /// Gets the size-like dimensions.
    ShapeRef getSizes() const { return m_sizes; }

    /// Obtains the default organization for this domain.
    AffineMap getDefaultOrganization(MLIRContext* ctx) const
    {
        return dlnn::getDefaultOrganization(ctx, 2 + getSizes().size(), false);
    }

    bool operator==(const WeightsDomain &) const = default;

    friend llvm::hash_code hash_value(const WeightsDomain &domain)
    {
        return llvm::hash_combine(
            domain.getNumInChannels(),
            domain.getNumOutChannels(),
            domain.getSizes());
    }

private:
    unsigned m_numInChannels;
    unsigned m_numOutChannels;
    ShapeRef m_sizes;
};

/// Computes the tensor shape for a FeatureMapDomain.
FailureOr<Shape> computeTensorShape(
    function_ref<InFlightDiagnostic()> emitError,
    const FeatureMapDomain &domain,
    AffineMap organization);

/// Computes the tensor shape for a WeightsDomain.
FailureOr<Shape> computeTensorShape(
    function_ref<InFlightDiagnostic()> emitError,
    const WeightsDomain &domain,
    AffineMap organization);

} // namespace mlir::dlnn
