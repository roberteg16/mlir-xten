/// Declaration of the DLNN dialect base.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#pragma once

#include "dlnn-mlir/Concepts/Concepts.h"
#include "dlnn-mlir/Dialect/DLNN/Enums.h"
#include "dlnn-mlir/Dialect/DLNN/Interfaces/EnclaveOpInterface.h"
#include "dlnn-mlir/Dialect/DLNN/Interfaces/GraphInterface.h"
#include "dlnn-mlir/Dialect/DLNN/Interfaces/NodeInterface.h"
#include "dlnn-mlir/Dialect/DLNN/Interfaces/ScalarInterface.h"
#include "dlnn-mlir/Dialect/DLNN/Interfaces/VolumeInterface.h"
#include "dlnn-mlir/Dialect/DLNN/Interfaces/VolumeOpInterface.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include <cstddef>
#include <cstdint>

namespace mlir::dlnn {

using namespace mlir::concepts;

//===----------------------------------------------------------------------===//
// Type aliases
//===----------------------------------------------------------------------===//
//
// Some common type aliases for the DLNN namespace, which clarify the intent of
// certain members and parameters.

/// Type that stores the size of a dimension.
using dim_size_t = std::decay_t<decltype(ShapedType::kDynamicSize)>;
/// Value that indicates a dynamically-sized dimension.
///
/// Since we are not inheriting from ShapedType, let's have a central definition
/// of this.
inline constexpr dim_size_t dynamic_size = ShapedType::kDynamicSize;

/// Type that hold a non-owning reference to a list of dimension sizes.
using ShapeRef = ArrayRef<dim_size_t>;
/// Parameter type that can be used to construct a list of dimension sizes.
using ShapeBuilder = SmallVectorImpl<dim_size_t>;
/// Type that stores a list of dimension sizes.
using Shape = SmallVector<dim_size_t>;

} // namespace mlir::dlnn

//===- Generated includes -------------------------------------------------===//

#include "dlnn-mlir/Dialect/DLNN/IR/Base.h.inc"

//===----------------------------------------------------------------------===//
