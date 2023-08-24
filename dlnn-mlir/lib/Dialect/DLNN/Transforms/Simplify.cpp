/// Implements the DLNN simplification pass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "PassDetail.h"
#include "dlnn-mlir/Dialect/DLNN/Utils/CommaSeparated.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/SideEffectUtils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dlnn-simplify"

using namespace llvm;
using namespace mlir;
using namespace mlir::dlnn;

namespace {

/// Removes unused capture arguments from EnclaveOp ops.
class RemoveUnusedCaptures : public OpInterfaceRewritePattern<EnclaveOp> {
public:
    using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

    virtual LogicalResult
    matchAndRewrite(EnclaveOp op, PatternRewriter &rewriter) const override
    {
        // Collect all unused block arguments.
        auto unused = to_vector(make_filter_range(
            op.getEnclaveBody().getArguments(),
            [](BlockArgument arg) { return arg.use_empty(); }));

        if (unused.empty()) return failure();

        // Update the op in-place.
        rewriter.updateRootInPlace(op, [&]() { op.uncapture(unused); });
        return success();
    }
};

/// Removes unused return values from EnclaveOp ops.
class RemoveUnusedReturns : public OpInterfaceRewritePattern<EnclaveOp> {
public:
    using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

    virtual LogicalResult
    matchAndRewrite(EnclaveOp op, PatternRewriter &rewriter) const override
    {
        // There must be results we can remove.
        if (!any_of(op->getResults(), [](OpResult result) {
                return result.use_empty();
            }))
            return failure();

        // NOTE: Results / Regions cannot erased / transferred in-place.
        // Clone the op
        OperationState state(op.getLoc(), op->getName());
        state.addOperands(op->getOperands());
        state.addAttributes(op->getAttrs());
        for (auto &region : op->getRegions()) {
            BlockAndValueMapping map;
            region.cloneInto(state.addRegion(), map);
        }

        // Build the new result list, remembering indices that were deleted.
        SmallVector<unsigned> indices;
        for (auto result : op->getResults())
            if (!result.use_empty())
                state.types.push_back(result.getType());
            else
                indices.push_back(result.getResultNumber());

        // Create the new op, and erase the deleted results from the terminator.
        rewriter.setInsertionPointAfter(op);
        auto newOp = cast<EnclaveOp>(rewriter.create(state));
        auto newTerminator = newOp.getTerminator();
        for (unsigned idx : reverse(indices)) newTerminator->eraseOperand(idx);

        // Replace the uses of the old op with the new op's results.
        unsigned newIndex = 0;
        for (auto result : op->getResults()) {
            if (result.use_empty()) continue;
            result.replaceAllUsesWith(newOp->getResult(newIndex++));
        }

        // The op was updated out-of-place.
        rewriter.eraseOp(op);
        return success();
    }
};

/// Removes graph embeddings that are unused and have no side-effects.
class RemoveDeadEmbeddings : public OpRewritePattern<EmbedOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    virtual LogicalResult
    matchAndRewrite(EmbedOp op, PatternRewriter &rewriter) const override
    {
        // All results must be unused.
        if (!op.getResults().use_empty()) return failure();

        // The graph must be side-effect free (which is defined by recursion).
        if (op.getGraphContent()
            && !all_of(
                op.getGraphContent()->without_terminator(),
                [](Operation &op) { return isSideEffectFree(&op); }))
            return failure();

        // The embedding can be removed.
        rewriter.eraseOp(op);
        return success();
    }
};

/// Removes graphs that have no uses.
///
/// BUG: This should automatically happen for symbols with non-public visiblity
///      or those that return true for canDiscardOnUseEmpty(). However, that
///      didn't work!
class RemoveDeadGraphs : public OpRewritePattern<GraphOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    virtual LogicalResult
    matchAndRewrite(GraphOp op, PatternRewriter &rewriter) const override
    {
        // Graph must be unused.
        if (!op.symbolKnownUseEmpty(op->getParentOp())) return failure();

        // The graph can be removed.
        rewriter.eraseOp(op);
        return success();
    }
};

} // namespace

void mlir::dlnn::populateSimplifyPatterns(RewritePatternSet &patterns)
{
    patterns.add<
        RemoveUnusedCaptures,
        RemoveUnusedReturns,
        RemoveDeadEmbeddings,
        RemoveDeadGraphs>(patterns.getContext());
}

namespace {

/// Simplifies DLNN networks.
class SimplifyPass : public DLNNSimplifyBase<SimplifyPass> {
public:
    using DLNNSimplifyBase::DLNNSimplifyBase;

    virtual void runOnOperation() override
    {
        RewritePatternSet patterns(&getContext());

        populateSimplifyPatterns(patterns);

        std::ignore = applyPatternsAndFoldGreedily(
            getOperation(),
            FrozenRewritePatternSet(std::move(patterns)));
    }
};

} // namespace

std::unique_ptr<Pass> mlir::dlnn::createSimplifyPass()
{
    return std::make_unique<SimplifyPass>();
}
