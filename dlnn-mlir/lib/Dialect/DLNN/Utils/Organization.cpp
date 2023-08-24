/// Implements utilities for working with organized types.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#include "dlnn-mlir/Dialect/DLNN/Utils/Organization.h"

#include "dlnn-mlir/Dialect/DLNN/Utils/CommaSeparated.h"
#include "dlnn-mlir/Dialect/DLNN/Utils/Presburger.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::dlnn;
using namespace mlir::presburger;

/// Gets the PresburgerSpace of a FeatureMapDomain.
///
/// Always has the form:
///
///     `(c,x_i,...)[N]`
static PresburgerSpace getSpace(const FeatureMapDomain &domain)
{
    return PresburgerSpace::getSetSpace(1 + domain.getSizes().size(), 1);
}
/// Gets the PresburgerSpace of a WeightsDomain.
///
/// Always has the form:
///
///     `(c,f,x_i,...)`
static PresburgerSpace getSpace(const WeightsDomain &domain)
{
    return PresburgerSpace::getSetSpace(2 + domain.getSizes().size());
}

/// Computes the IntegerPolyhedron of all valid points in the domain described
/// by @p space , @p channels and @p sizes .
///
/// @pre        @p space has enough dims to accomodate all dims of @p channels
///             and @p sizes
/// @pre        @p channels has no dynamic or empty dimensions.
/// @pre        @p sizes has no empty dimensions.
static IntegerPolyhedron
getDomain(const PresburgerSpace &space, ShapeRef channels, ShapeRef sizes)
{
    assert(space.getNumDimIds() == channels.size() + sizes.size());

    // Reserve the matrix.
    //
    // Each static dimension will have two bounds, and each dynamic symbol will
    // have one.
    IntegerPolyhedron result(
        space.getNumDimIds() * 2 + space.getNumSymbolIds(),
        0,
        space.getNumIds() + 1,
        space);

    // Apply the static bounds.
    {
        unsigned dim = 0;
        for (auto ch : channels) {
            assert(ch > 0);
            result.addBound(IntegerPolyhedron::BoundType::LB, dim, 0);
            result.addBound(IntegerPolyhedron::BoundType::UB, dim++, ch - 1);
        }
        for (auto sz : sizes) {
            assert(sz > 0 || sz == dynamic_size);
            result.addBound(IntegerPolyhedron::BoundType::LB, dim, 0);
            if (sz != dynamic_size)
                result.addBound(IntegerPolyhedron::BoundType::UB, dim, sz - 1);
            ++dim;
        }
    }

    // Apply dynamic bounds.
    {
        const auto offset = space.getIdKindOffset(IdKind::Symbol);
        for (auto sym : iota_range<unsigned>(0, space.getNumSymbolIds(), false))
            result.addBound(IntegerPolyhedron::BoundType::LB, offset + sym, 0);
    }

    return result;
}
/// Computes the IntegerPolyhedron of all valid points in @p domain .
static IntegerPolyhedron getDomain(const FeatureMapDomain &domain)
{
    return getDomain(
        getSpace(domain),
        {domain.getNumChannels()},
        domain.getSizes());
}
/// Computes the IntegerPolyhedron of all valid points in @p domain .
static IntegerPolyhedron getDomain(const WeightsDomain &domain)
{
    return getDomain(
        getSpace(domain),
        {domain.getNumInChannels(), domain.getNumOutChannels()},
        domain.getSizes());
}

static FailureOr<Shape> computeTensorShape(
    mlir::function_ref<InFlightDiagnostic()> emitError,
    IntegerPolyhedron inDomain,
    AffineMap organization)
{
    // Flatten the organization into a Map.
    if (!organization) return emitError() << "expected organization map";
    auto maybeOrgMap = Map::fromAffineMap(organization);
    if (!maybeOrgMap) return emitError() << "organization is not pure-affine";
    const auto inRank = inDomain.getNumDimIds();
    const auto outRank = maybeOrgMap->getNumRangeIds();

    // Example:
    // organization:
    //     (C, W, H)[N] -> (H, C floordiv 8, W, N, C mod 8)
    //
    // orgMap:
    //     | C  W  H | y0 y1 y2 y3 y4 |  N | l0 |  c      |
    //     | ------- | -------------- | -- | -- | ------- |
    //     | 0  0  1 | -1  0  0  0  0 |  0 |  0 |  0 = 0  | y0 = H
    //     | 0  0  0 |  0 -1  0  0  0 |  0 |  1 |  0 = 0  | y1 = l0
    //     | 0  1  0 |  0  0 -1  0  0 |  0 |  0 |  0 = 0  | y2 = W
    //     | 0  0  0 |  0  0  0 -1  0 |  1 |  0 |  0 = 0  | y3 = N
    //     | 1  0  0 |  0  0  0  0 -1 |  0 | -8 |  0 = 0  | y4 = C - 8l0
    //     | 1  0  0 |  0  0  0  0  0 |  0 | -8 |  0 >= 0 | C >= 8l0
    //     | -1 0  0 |  0  0  0  0  0 |  0 |  8 |  7 >= 0 | C <= 8l0 + 7

    // Constrain the forward map to the input domain.
    Map fwdMap(*maybeOrgMap);
    if (fwdMap.getNumSymbolIds() != inDomain.getNumSymbolIds())
        return emitError()
               << "organization symbols (" << fwdMap.getNumSymbolIds()
               << ") != type symbols (" << inDomain.getNumSymbolIds() << ")";
    if (fwdMap.getNumDomainIds() != inRank)
        return emitError()
               << "organization domain dims (" << fwdMap.getNumDomainIds()
               << ") != type dims (" << inDomain.getNumDimIds() << ")";
    fwdMap.intersectDomain(inDomain);

    // Compute the shape of the output tensor.
    // NOTE: This strategy uses a very simple lookup through inequalities, which
    //       may falesly return unbounded for more complex constraint sets.
    Shape outShape(outRank, dynamic_size);
    for (auto outDim : iota_range<unsigned>(0, outRank, false)) {
        auto maybeUB =
            fwdMap.getConstantBound(Map::BoundType::UB, inRank + outDim);
        if (maybeUB.hasValue()) {
            assert(maybeUB.getValue() >= 0);
            outShape[outDim] = maybeUB.getValue() + 1;
        }
    }

    // Construct a domain set from the output shape.
    const auto outDomain = getDomain(
        PresburgerSpace::getSetSpace(outRank, fwdMap.getNumSymbolIds()),
        {},
        outShape);

    // Constrain the backward map to the output domain.
    Map bwdMap(*maybeOrgMap);
    bwdMap.intersectRange(outDomain);

    // Check that there is no implicit padding.
    Shape sample;
    if (PresburgerSet(bwdMap.getDomainSet())
            .subtract(PresburgerSet(inDomain))
            .coalesce()
            .findIntegerSample(sample)) {
        sample.resize(inRank);
        return emitError()
               << "organization is not surjective over inferred tensor domain ["
               << CommaSeparated(outShape) << "] (e.g. missing ["
               << CommaSeparated(sample) << "])";
    }

    return std::move(outShape);
}

FailureOr<Shape> mlir::dlnn::computeTensorShape(
    function_ref<InFlightDiagnostic()> emitError,
    const FeatureMapDomain &domain,
    AffineMap organization)
{
    return ::computeTensorShape(emitError, getDomain(domain), organization);
}

FailureOr<Shape> mlir::dlnn::computeTensorShape(
    function_ref<InFlightDiagnostic()> emitError,
    const WeightsDomain &domain,
    AffineMap organization)
{
    return ::computeTensorShape(emitError, getDomain(domain), organization);
}
