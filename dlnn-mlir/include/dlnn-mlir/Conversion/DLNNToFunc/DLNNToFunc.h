/// Declares the DLNNToFunc pass entry point.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Support/LLVM.h"

#include <memory>

namespace mlir {

class Pass;
class RewritePatternSet;

void populateDLNNNetworkToFuncPatterns(RewritePatternSet &patterns);
void populateDLNNGraphToFuncPatterns(RewritePatternSet &patterns);

/// Creates the dlnn to func conversion pass.
std::unique_ptr<Pass> createConvertDLNNToFuncPass();

} // namespace mlir
