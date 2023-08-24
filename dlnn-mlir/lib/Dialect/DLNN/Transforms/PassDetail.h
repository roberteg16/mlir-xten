/// Private include for pass implementations.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "dlnn-mlir/Dialect/DLNN/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::dlnn {

#define GEN_PASS_CLASSES
#include "dlnn-mlir/Dialect/DLNN/Transforms/Passes.h.inc"

} // namespace mlir::dlnn
