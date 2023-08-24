/// Declares the DLNN dialect enums.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir::dlnn {

/// Enumeration of known scalar operations.
///
/// This is a non-exhaustive list of supported scalar operations that we expect
/// to appear in DLNN graphs. This enumeration exists primarily to:
///     (1) Classify operations on scalars in generic operator nodes.
///     (2) Define the ScalarInterface, which materializes required ops.
///
/// This list was initially populated with the ONNX scalar operators:
///     https://github.com/onnx/onnx/blob/main/docs/Operators.md
enum class ScalarOpKind {
    /// Unknown scalar operation.
    Unknown = 0,

    Abs,
    Acos,
    Acosh,
    Add,
    And,
    Asin,
    Asinh,
    Atan,
    Atanh,
    Cos,
    Cosh,
    Ceil,
    Div,
    Erf,
    Exp,
    Floor,
    Log,
    Max,
    Min,
    Mod,
    Mul,
    Neg,
    Not,
    Or,
    Pow,
    Reciprocal,
    Round,
    Sin,
    Sinh,
    Sub,
    Tan,
    Tanh,
    Xor
};

} // namespace mlir::dlnn

//===- Generated includes -------------------------------------------------===//

#include "dlnn-mlir/Dialect/DLNN/Enums.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::dlnn {

// raw_ostream& operator<<(...)

} // namespace mlir::dlnn

namespace mlir {

// template<class> FieldParser<...>

} // namespace mlir
