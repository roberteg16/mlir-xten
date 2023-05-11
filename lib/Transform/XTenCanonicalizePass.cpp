#include "xten/Transform/XTenCanonicalizePass.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#define DEBUG_TYPE "xten-canonicalize"

namespace xilinx::xten {
using namespace mlir;
#define GEN_PASS_DEF_XTENCANONICALIZEPASS
#include "xten/Transform/Passes.h.inc"
} // namespace xilinx::xten

using namespace mlir;
using namespace xilinx;
using namespace xilinx::xten;

namespace {
struct XTenCanonicalizePass
    : public xten::impl::XTenCanonicalizePassBase<XTenCanonicalizePass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    populateXTenCanonicalizePatterns(patterns);

    if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))
            .failed())
      signalPassFailure();
  }
};
} // namespace

void xilinx::xten::populateXTenCanonicalizePatterns(
    RewritePatternSet &patterns) {
  populateXTenFoldConcatPatterns(patterns);
}

std::unique_ptr<Pass> xilinx::xten::createXTenCanonicalizePass() {
  return std::make_unique<XTenCanonicalizePass>();
}
