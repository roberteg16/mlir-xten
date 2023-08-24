/// Implements the DLNN dialect volume op interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#include "dlnn-mlir/Dialect/DLNN/Interfaces/VolumeOpInterface.h"

#include "dlnn-mlir/Dialect/DLNN/IR/DLNN.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::dlnn;
using namespace mlir::presburger;

//===- Generated implementation -------------------------------------------===//

#include "dlnn-mlir/Dialect/DLNN/Interfaces/VolumeOpInterface.cpp.inc"

//===----------------------------------------------------------------------===//

void VolumeAccessMap::dump() const
{
    errs() << "Access map for " << getVolume() << " (" << getAccess() << "):\n";
    getMap().dump();
}

void ComprehensionSpace::dump() const
{
    errs() << "ComprehensionSpace(" << getRank() << ", " << getNumTrailing()
           << ")\n";
}

VolumeRelation::VolumeRelation(
    OpTiedValue source,
    VolumeAccess access,
    const ComprehensionSpace &domainSpace)
        : vector(),
          m_domainSpace(domainSpace)
{
    // Create the sourceMap.
    const auto space = Space::getRelationSpace(
        domainSpace.getNumDimIds(),
        domainSpace.getRank());
    Map sourceMap(0, domainSpace.getRank(), space.getNumIds() + 1, space);

    // Initialize the sourceMap with identity.
    const auto rangeOffset = space.getIdKindOffset(IdKind::Range);
    SmallVector<int64_t> coeff(sourceMap.getNumCols(), 0);
    for (auto dim : iota_range<unsigned>(0, domainSpace.getRank(), false)) {
        coeff[dim] = -1;
        coeff[rangeOffset + dim] = 1;
        sourceMap.addEquality(coeff);
        coeff[dim] = coeff[rangeOffset + dim] = 0;
    }

    // Put the sourceMap at the front.
    vector::emplace_back(source, access, std::move(sourceMap));
}

/// Gets the indexing map associated with @p volume .
static AffineMap getTiedIndexingMap(linalg::LinalgOp op, OpTiedValue volume)
{
    assert(volume.getOwner() == op);

    if (auto res = volume.dyn_cast<OpResult>())
        return op.getTiedIndexingMapForResult(res);
    return op.getTiedIndexingMap(volume.get<OpOperand*>());
}

/// Gets the VolumeAccess associated with @p volume .
static VolumeAccess getTiedAccess(linalg::LinalgOp op, OpTiedValue volume)
{
    assert(volume.getOwner() == op);

    // Result values can only be written.
    if (volume.is<OpResult>()) return VolumeAccess::Write;
    const auto operand = volume.get<OpOperand*>();

    // A non-output operand is only ever read.
    if (!is_contained(op.getOutputOperands(), operand))
        return VolumeAccess::Read;

    // Tensor operands can't be written to either!
    if (volume.getValue().getType().isa<TensorType>())
        return VolumeAccess::Read;

    // Otherwise, usage depends on usage within the payload region.
    return op.payloadUsesValueFromOperand(operand) ? VolumeAccess::ReadWrite
                                                   : VolumeAccess::Write;
}

/// Gets the VolumeRelation of @p source for a linalg::LinalgOp @p op .
static std::optional<VolumeRelation>
getVolumeRelation(linalg::LinalgOp op, OpTiedValue source)
{
    assert(op && source);
    assert(source.getOwner() == op);

    auto maybeSourceMap = Map::fromAffineMap(getTiedIndexingMap(op, source));
    if (!maybeSourceMap) return std::nullopt;

    // Turn the source index map into the source VolumeAccessMap, creating the
    // synthetic iteration space.
    const auto sourceAccess = getTiedAccess(op, source);
    const auto numTrailing = maybeSourceMap->promoteIndependentRangeIds();
    const ComprehensionSpace space(
        maybeSourceMap->getNumDomainIds() - numTrailing,
        numTrailing);
    maybeSourceMap->inverse();

    VolumeRelation result(source, sourceAccess, space);
    // Give a regular indexing map, project it to the synthetic iteration
    // space and insert it into the result.
    const auto unite = [&](OpTiedValue volume, const Map &map) {
        Map part(*maybeSourceMap);
        part.compose(map);
        part.removeRedundantLocalVars();
        result.unite(VolumeAccessMap(
            volume,
            getTiedAccess(op, volume),
            std::move(part)));
    };
    // Processes volume and inserts it into the result, indicating failure.
    const auto insert = [&](OpTiedValue volume) -> bool {
        if (volume == source) return true;
        auto maybeMap = Map::fromAffineMap(getTiedIndexingMap(op, volume));
        if (!maybeMap) return false;
        unite(volume, *maybeMap);
        return true;
    };

    // Populate the result.
    for (auto tied : op.getInputOperands())
        if (!insert(tied)) return std::nullopt;
    for (auto tied : op.getOutputOperands())
        if (!insert(tied)) return std::nullopt;
    for (auto tied : op->getOpResults())
        if (!insert(tied)) return std::nullopt;

    return result;
}

SetAndSymbolValues VolumeRelation::getIterationDomain(
    VolumeDomainMap &volumeDomains,
    bool allowResults) const
{
    // Create the universe iteration domain.
    Set result(
        getDomainSpace().getNumDimIds() * 2,
        0,
        getDomainSpace().getNumIds() + 1,
        getDomainSpace());
    SmallVector<Value> symbolValues;

    // Incrementally constrain the iteration domain using the volume domains.
    for (auto &map : *this) {
        if (map.getVolume().is<OpResult>() && !allowResults) continue;

        auto maybeDomain = volumeDomains.getOrCompute(map.getVolume());
        if (!maybeDomain) continue;

        const auto numSymbols = maybeDomain->getSet().getNumSymbolIds();

        // Add the required symbols to the existing map.
        Map copy(map);
        copy.appendId(IdKind::Symbol, numSymbols);
        // Intersect with the volume domain to obtain the iteration space range.
        copy.intersectRange(*maybeDomain);
        auto localDomain = Set(copy.getDomainSet());
        localDomain.removeRedundantLocalVars();

        // Align the symbol ids for both sets, remvoing duplicates.
        const auto oldSymbolOffset =
            localDomain.getIdKindOffset(IdKind::Symbol);
        auto newSymbolOffset = oldSymbolOffset + symbolValues.size();
        localDomain.insertId(IdKind::Symbol, 0, result.getNumSymbolIds());
        for (auto idx : iota_range<unsigned>(0, numSymbols, false)) {
            const auto val = maybeDomain->getSymbolValue(idx);
            const auto it = llvm::find(symbolValues, val);
            if (it == symbolValues.end()) {
                symbolValues.push_back(val);
                result.appendId(IdKind::Symbol);
                continue;
            }
            const auto with = std::distance(symbolValues.begin(), it);
            localDomain.swapId(newSymbolOffset + idx, oldSymbolOffset + with);
            localDomain.removeId(newSymbolOffset + idx);
        }

        // Intersect the result set with the local domain.
        result.append(localDomain);
    }

    // Simplify the result set.
    result.removeTrivialRedundancy();
    result.removeRedundantInequalities();
    result.removeRedundantLocalVars();
    result.removeRedundantConstraints();

    return SetAndSymbolValues(std::move(result), std::move(symbolValues));
}

void VolumeRelation::dump() const
{
    errs() << "Volume relation for " << getSource() << ":\n";
    getDomainSpace().dump();
    errs() << "\n";
    for (auto &map : *this) {
        errs() << "Access map for ";
        if (auto res = map.getVolume().dyn_cast<OpResult>())
            errs() << "result #" << res.getResultNumber();
        else
            errs() << "operand #"
                   << map.getVolume().get<OpOperand*>()->getOperandNumber();
        errs() << ":\n";
        map.getMap().dump();
    }
}

std::optional<VolumeRelation> mlir::dlnn::getVolumeRelation(OpTiedValue source)
{
    assert(source);

    // Delegate to the owning operation.
    const auto op = source.getOwner();
    if (auto iface = dyn_cast<VolumeOpInterface>(op))
        return iface.getVolumeRelation(source);
    if (auto iface = dyn_cast<linalg::LinalgOp>(op))
        return ::getVolumeRelation(iface, source);

    // No fallback available.
    return std::nullopt;
}
