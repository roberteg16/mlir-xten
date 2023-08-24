/// Declares the DLNN pass entry points.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "dlnn-mlir/Dialect/DLNN/IR/DLNN.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

class RewritePatternSet;

} // namespace mlir

namespace mlir::dlnn {

/// Obtains the graph flattening patterns.
void populateFlattenPatterns(RewritePatternSet &patterns);

/// Creates the graph flattening pass.
std::unique_ptr<Pass> createFlattenPass();

/// Obtains the graph simplification patterns.
void populateSimplifyPatterns(RewritePatternSet &patterns);

/// Creates the graph simplification pass.
std::unique_ptr<Pass> createSimplifyPass();

/// Creates the tiling pass.
std::unique_ptr<Pass> createApplyTilingPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "dlnn-mlir/Dialect/DLNN/Transforms/Passes.h.inc"

} // namespace mlir::dlnn
