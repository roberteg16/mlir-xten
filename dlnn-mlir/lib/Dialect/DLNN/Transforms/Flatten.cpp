/// Implements the DLNN flattening pass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "PassDetail.h"
#include "dlnn-mlir/Dialect/DLNN/Utils/CommaSeparated.h"
#include "dlnn-mlir/Dialect/DLNN/Utils/STLExtras.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dlnn-flatten"

using namespace llvm;
using namespace mlir;
using namespace mlir::dlnn;

namespace {

/// Flattens a SubgraphOp (essentially inlining it).
class FlattenSubgraph : public OpRewritePattern<SubgraphOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    virtual LogicalResult
    matchAndRewrite(SubgraphOp op, PatternRewriter &rewriter) const override
    {
        auto &body = op.getEnclaveBody();

        BlockAndValueMapping map;
        for (auto [idx, capture] : enumerate(op.getCaptures()))
            map.map(body.getArgument(idx), capture);
        for (auto &op : body.without_terminator())
            rewriter.insert(op.clone(map));
        for (auto [idx, result] : enumerate(op.getResults()))
            result.replaceAllUsesWith(
                map.lookup<Value>(body.getTerminator()->getOperand(idx)));

        rewriter.eraseOp(op);
        return success();
    }
};

} // namespace

void mlir::dlnn::populateFlattenPatterns(RewritePatternSet &patterns)
{
    patterns.add<FlattenSubgraph>(patterns.getContext());
}

namespace {

/// Flattens DLNN graphs.
class FlattenPass : public DLNNFlattenBase<FlattenPass> {
public:
    using DLNNFlattenBase::DLNNFlattenBase;

    virtual void runOnOperation() override
    {
        RewritePatternSet patterns(&getContext());

        populateFlattenPatterns(patterns);

        std::ignore = applyPatternsAndFoldGreedily(
            getOperation(),
            FrozenRewritePatternSet(std::move(patterns)));
    }
};

} // namespace

std::unique_ptr<Pass> mlir::dlnn::createFlattenPass()
{
    return std::make_unique<FlattenPass>();
}
