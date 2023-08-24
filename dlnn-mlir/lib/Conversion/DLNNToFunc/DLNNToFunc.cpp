/// Implements the DLNNToFunc pass.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "dlnn-mlir/Conversion/DLNNToFunc/DLNNToFunc.h"

#include "../PassDetail.h"
#include "dlnn-mlir/Dialect/DLNN/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::dlnn;

namespace {

class GraphToFunc : public OpRewritePattern<GraphOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    virtual LogicalResult
    matchAndRewrite(GraphOp op, PatternRewriter &rewriter) const override
    {
        // We insert the resulting symbol next to the network of this graph.
        auto network = op->getParentOfType<NetworkOp>();
        rewriter.setInsertionPointAfter(network);

        // Make a unique name for this graph.
        auto name = rewriter.getStringAttr(
            Twine(network.getSymName()).concat("_").concat(op.getSymName()));

        // Create a FuncOp for this graph.
        auto funcOp = rewriter.create<func::FuncOp>(
            op.getLoc(),
            name,
            op.getFunctionType());
        funcOp.setVisibility(SymbolTable::Visibility::Private);

        // Move the graph contents to the FuncOp.
        funcOp.getBody().getBlocks().splice(
            funcOp.getBody().getBlocks().begin(),
            op.getBodyRegion().getBlocks());

        // Replace all uses of the old symbol.
        if (failed(op.replaceAllSymbolUses(name, network))) return failure();

        // Replace the OutputOp with a ReturnOp.
        rewriter.setInsertionPoint(&funcOp.getBody().back().back());
        rewriter.replaceOpWithNewOp<func::ReturnOp>(
            &funcOp.getBody().back().back(),
            funcOp.getBody().back().back().getOperands());

        // Erase the old GraphOp.
        rewriter.eraseOp(op);
        return success();
    }
};

class EmbedToCall : public OpRewritePattern<EmbedOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    virtual LogicalResult
    matchAndRewrite(EmbedOp op, PatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<func::CallOp>(
            op,
            op.getGraphRef(),
            op.getResultTypes(),
            op.getOperands());
        return success();
    }
};

} // namespace

void mlir::populateDLNNGraphToFuncPatterns(RewritePatternSet &patterns)
{
    patterns.add<GraphToFunc, EmbedToCall>(patterns.getContext());
    populateFlattenPatterns(patterns);
}

namespace {

class NetworkToFunc : public OpRewritePattern<NetworkOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    virtual LogicalResult
    matchAndRewrite(NetworkOp op, PatternRewriter &rewriter) const override
    {
        // Create a FuncOp for this network.
        auto funcOp = rewriter.create<func::FuncOp>(
            op.getLoc(),
            op.getName(),
            op.getFunctionType());

        // Move the network contents to the FuncOp.
        funcOp.getBody().getBlocks().splice(
            funcOp.getBody().getBlocks().begin(),
            op.getBodyRegion().getBlocks());

        // Replace the OutputOp with a ReturnOp.
        rewriter.setInsertionPoint(&funcOp.getBody().back().back());
        rewriter.replaceOpWithNewOp<func::ReturnOp>(
            &funcOp.getBody().back().back(),
            funcOp.getBody().back().back().getOperands());

        // Erase the old NetworkOp.
        rewriter.eraseOp(op);
        return success();
    }
};

class EvalToCall : public OpRewritePattern<EvalOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    virtual LogicalResult
    matchAndRewrite(EvalOp op, PatternRewriter &rewriter) const override
    {
        rewriter.replaceOpWithNewOp<func::CallOp>(
            op,
            op.getNetworkRef(),
            op.getResultTypes(),
            op.getOperands());
        return success();
    }
};

} // namespace

void mlir::populateDLNNNetworkToFuncPatterns(RewritePatternSet &patterns)
{
    patterns.add<NetworkToFunc, EvalToCall>(patterns.getContext());
}

namespace {

/// Converts DLNN networks and graphs to functions.
class DLNNToFuncPass : public ConvertDLNNToFuncBase<DLNNToFuncPass> {
public:
    using ConvertDLNNToFuncBase::ConvertDLNNToFuncBase;

    virtual void runOnOperation() override
    {
        if (failed(graphToFunc()) || failed(networkToFunc()))
            signalPassFailure();
    }

private:
    LogicalResult graphToFunc()
    {
        ConversionTarget target(getContext());

        target.addIllegalOp<GraphOp>();
        target.addIllegalOp<EmbedOp>();
        target.addLegalDialect<func::FuncDialect>();
        target.addLegalOp<func::FuncOp>();

        RewritePatternSet patterns(&getContext());

        populateDLNNGraphToFuncPatterns(patterns);

        return applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns));
    }

    LogicalResult networkToFunc()
    {
        ConversionTarget target(getContext());

        target.addIllegalOp<NetworkOp>();
        target.addIllegalOp<EvalOp>();
        target.addLegalDialect<func::FuncDialect>();
        target.addLegalOp<func::FuncOp>();

        RewritePatternSet patterns(&getContext());

        populateDLNNNetworkToFuncPatterns(patterns);

        return applyPartialConversion(
            getOperation(),
            target,
            std::move(patterns));
    }
};

} // namespace

std::unique_ptr<Pass> mlir::createConvertDLNNToFuncPass()
{
    return std::make_unique<DLNNToFuncPass>();
}
