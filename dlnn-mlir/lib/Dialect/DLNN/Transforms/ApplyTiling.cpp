#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dlnn-apply-tiling"

using namespace llvm;
using namespace mlir;
using namespace mlir::dlnn;

static OpFoldResult getConstantIndexOrValue(Value value)
{
    if (auto op = value.getDefiningOp<arith::ConstantIndexOp>())
        return op.getValue();

    return value;
}

static OpFoldResult buildSymbolExpr(
    OpBuilder &builder,
    Location loc,
    ArrayRef<int64_t> coeffs,
    ArrayRef<Value> symbolValues)
{
    assert(
        symbolValues.size() == coeffs.size() - 1
        && "mismatched number of symbols");

    // Turn the coefficients into an AffineExpr.
    auto expr = getAffineExprFromFlatForm(
        coeffs,
        0,
        symbolValues.size(),
        {},
        builder.getContext());
    // Turn the expression into a single-result AffineMap.
    auto map = AffineMap::get(0, symbolValues.size(), expr);
    // Evaluate the map.
    return getConstantIndexOrValue(
        builder.createOrFold<AffineApplyOp>(loc, map, symbolValues));
}

static SmallVector<OpFoldResult>
buildSizes(OpBuilder &builder, Location loc, const SetAndSymbolValues &domain)
{
    SmallVector<OpFoldResult> result(domain.getSet().getNumDimIds(), {});

    Set peeled(domain.getSet());
    peeled.projectOut(
        peeled.getIdKindOffset(IdKind::Local),
        peeled.getNumLocalIds());
    auto maybeBounds = peeled.peelBounds();
    if (!maybeBounds || !maybeBounds->isBounded()) return result;

    SmallVector<int64_t> coeff(domain.getSymbolValues().size() + 1);
    for (auto dim : iota_range<unsigned>(0, result.size(), false)) {
        copy(*maybeBounds->getUpper(dim), coeff.begin());
        for (auto [idx, val] : enumerate(*maybeBounds->getLower(dim)))
            coeff[idx] -= val;
        coeff.back() += 1;

        result[dim] =
            buildSymbolExpr(builder, loc, coeff, domain.getSymbolValues());
    }

    return result;
}

namespace {

inline constexpr StringRef tileSizesAttrName = "dlnn.tile_sizes";

class ApplyTilingPass : public ApplyTilingPassBase<ApplyTilingPass> {
public:
    using ApplyTilingPassBase::ApplyTilingPassBase;

    void runOnOperation() override
    {
        getOperation().walk([&](Operation* op, const WalkStage &) {
            auto attr = op->getAttr(tileSizesAttrName);
            if (!attr) return WalkResult::advance();
            auto arrayAttr = attr.dyn_cast<ArrayAttr>();
            if (!arrayAttr) {
                op->emitOpError("invalid `dlnn.tile_sizes` attribute");
                signalPassFailure();
                return WalkResult::interrupt();
            }
            if (op->getNumResults() != 1) {
                op->emitOpError("expected single result");
                signalPassFailure();
                return WalkResult::interrupt();
            }

            auto tileSizes = to_vector(map_range(
                arrayAttr.getAsValueRange<IntegerAttr>(),
                [](auto &&apint) { return apint.getSExtValue(); }));

            if (all_of(
                    tileSizes,
                    [](auto x) { return x == 0; })) {
                op->emitWarning("no tile sizes specified");
                return WalkResult::skip();
            }

            auto maybeRelation = getVolumeRelation(op->getOpResult(0));
            if (!maybeRelation) {
                op->emitError("failed to construct VolumeRelation");
                signalPassFailure();
                return WalkResult::interrupt();
            }

            if (failed(applyTiling(*maybeRelation, tileSizes))) {
                signalPassFailure();
                return WalkResult::interrupt();
            }
            return WalkResult::skip();
        });
    }

private:
    LogicalResult
    applyTiling(VolumeRelation &relation, ShapeRef sourceTileSizes)
    {
        const auto sourceOp = relation.getSourceOp();

        // Infer the iteration domain.
        const auto iterationDomain = relation.getIterationDomain();
        const auto numDynamics = iterationDomain.getSymbolValues().size();
        const auto maybeIVs = iterationDomain.getSet().toOffsetsStridesSizes();
        if (!maybeIVs)
            return sourceOp->emitOpError("failed to infer iteration domain");

        // Expand the tiling space to the new dimensions and get the map.
        auto newTileSizes = to_vector(sourceTileSizes);
        const auto numNewDims = relation.getDomainSpace().getNumTrailing();
        newTileSizes.append(numNewDims, 0);
        const auto tilingMap = Map::forTiling(newTileSizes);
        const auto numTileIVs = tilingMap.getNumSymbolIds();

        OpBuilder builder(sourceOp);

        // Create a list of all volumes that will be written to.
        const auto resultVolumes = to_vector(map_range(
            make_filter_range(
                relation,
                [](auto &map) {
                    return map.getVolume().template is<OpResult>();
                }),
            [&](const VolumeAccessMap &map) {
                Map copy(map.getMap());
                copy.insertId(
                    IdKind::Symbol,
                    0,
                    iterationDomain.getSymbolValues().size());
                copy.intersectDomain(iterationDomain.getSet());
                auto range = Set(copy.getRangeSet());

                return createUndefinedVolume(
                    builder,
                    map.getVolume().getType(),
                    buildSizes(
                        builder,
                        sourceOp->getLoc(),
                        SetAndSymbolValues(
                            range,
                            to_vector(iterationDomain.getSymbolValues()))),
                    sourceOp->getLoc());
            }));

        // Apply the tiling map to all volumes.
        relation.map([&](Map &map) {
            // Extent the map with the tile index IVs.
            map.insertId(IdKind::Symbol, 0, numTileIVs);

            // Construct a new map that operates on the tiled index space.
            Map tiledMap(tilingMap);
            tiledMap.compose(map);
            tiledMap.removeRedundantLocalVars();
            map = std::move(tiledMap);
        });

        // Prepare the tiled iteration domain.
        Set tiledIterationDomain(iterationDomain);
        tiledIterationDomain.insertId(IdKind::Symbol, 0, numTileIVs);
        SmallVector<Value> symbolValues;

        // Construct the loop nest.
        auto zero =
            builder.create<arith::ConstantIndexOp>(sourceOp->getLoc(), 0);
        auto one =
            builder.create<arith::ConstantIndexOp>(sourceOp->getLoc(), 1);
        SmallVector<scf::ForOp> tileLoops;
        for (auto [oldIV, sz] : enumerate(sourceTileSizes)) {
            if (sz == 0) continue;

            auto oldSize = builder.createOrFold<AffineApplyOp>(
                sourceOp->getLoc(),
                AffineMap::get(
                    0,
                    numDynamics,
                    getAffineExprFromFlatForm(
                        maybeIVs->getSize(oldIV),
                        0,
                        numDynamics,
                        {},
                        builder.getContext())),
                iterationDomain.getSymbolValues());
            auto newSize = builder.createOrFold<arith::FloorDivSIOp>(
                sourceOp->getLoc(),
                oldSize,
                builder.create<arith::ConstantIndexOp>(sourceOp->getLoc(), sz));

            auto args = tileLoops.empty()
                            ? ValueRange(resultVolumes)
                            : ValueRange(tileLoops.back().getRegionIterArgs());

            tileLoops.push_back(builder.create<scf::ForOp>(
                sourceOp->getLoc(),
                zero,
                newSize,
                one,
                args,
                [&](OpBuilder, Location, Value, ValueRange) {}));

            builder.setInsertionPointToStart(tileLoops.back().getBody());
            symbolValues.push_back(tileLoops.back().getInductionVar());
        }
        symbolValues.append(
            iterationDomain.getSymbolValues().begin(),
            iterationDomain.getSymbolValues().end());

        // Compute all the tile parameters.
        builder.setInsertionPointToStart(tileLoops.back().getBody());
        llvm::DenseMap<OpTiedValue, HyperRectangle> tileParams;
        for (auto &map : relation) {
            // Get the subvolume set.
            Map copy(map.getMap());
            copy.appendId(IdKind::Symbol, numDynamics);
            copy.intersectDomain(tiledIterationDomain);
            auto subvolumeIndices = Set(copy.getRangeSet());
            subvolumeIndices.removeIdRange(IdKind::Symbol, 0, numTileIVs);

            // Extract the tile parameters.
            auto maybeTile = subvolumeIndices.toOffsetsStridesSizes();
            if (!maybeTile || !maybeTile->isDefinite()) {
                return sourceOp->emitOpError(
                           "failed to infer tile parameters for ")
                       << map.getVolume().getValue();
            }
            maybeTile->stripLocals();

            // Insert the tile IV symbols and strip the locals.
            maybeTile->insertSymbols(0, numTileIVs);
            for (auto dim :
                 iota_range<unsigned>(0, maybeTile->getRank(), false)) {
                unsigned tileIV = 0;
                for (auto sz : sourceTileSizes) {
                    if (sz == 0) continue;
                    maybeTile->getOffset(dim)[tileIV++] = sz;
                }
            }

            // Store them in the map for later.
            tileParams.try_emplace(
                map.getVolume(),
                HyperRectangle::fromOffsetStridesSizes(
                    builder,
                    sourceOp->getLoc(),
                    *maybeTile,
                    symbolValues));
        }

        // Make a remapping for all inputs.
        BlockAndValueMapping remap;
        for (auto &map : relation) {
            if (!map.isRead()) continue;
            auto read = extractSubvolume(
                builder,
                map.getVolume().getValue(),
                tileParams.find(map.getVolume())->getSecond());
            remap.map(map.getVolume().getValue(), read);
        }

        // Clone the operation in question.
        SmallVector<Value> tiledResults;
        tiledResults.reserve(sourceOp->getNumResults());
        auto targetOp = builder.clone(*sourceOp, remap);
        for (auto &map : relation) {
            if (!map.isWrite()) continue;

            auto volume =
                tileLoops.front().getIterOperands()
                    [map.getVolume().get<OpResult>().getResultNumber()];

            auto &tileParam = tileParams.find(map.getVolume())->getSecond();
            auto result = targetOp->getResult(
                map.getVolume().get<OpResult>().getResultNumber());

            result.setType(getSubvolumeType(result.getType(), tileParam));
            tiledResults.push_back(
                insertSubvolume(builder, volume, tileParam, result));
        }
        builder.create<scf::YieldOp>(targetOp->getLoc(), tiledResults);

        // Update the loop yield values.
        for (auto loopIdx :
             reverse(iota_range<unsigned>(0, tileLoops.size() - 1, false))) {
            auto &loop = tileLoops[loopIdx];
            builder.setInsertionPointToEnd(loop.getBody());
            builder.create<scf::YieldOp>(
                sourceOp->getLoc(),
                tileLoops[loopIdx + 1].getResults());
        }

        // Replace the old operation.
        sourceOp->replaceAllUsesWith(tileLoops.front().getResults());
        sourceOp->erase();

        return success();
    }
};

} // namespace

std::unique_ptr<Pass> mlir::dlnn::createApplyTilingPass()
{
    return std::make_unique<ApplyTilingPass>();
}
