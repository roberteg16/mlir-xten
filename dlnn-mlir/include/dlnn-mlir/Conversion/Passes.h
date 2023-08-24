/// Declares the DLNN conversion pass entry points.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "dlnn-mlir/Conversion/DLNNToFunc/DLNNToFunc.h"
#include "mlir/Pass/Pass.h"

namespace mlir::dlnn {

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "dlnn-mlir/Conversion/Passes.h.inc"

} // namespace mlir::dlnn
