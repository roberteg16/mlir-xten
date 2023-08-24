/// Declares the DLNN scalar interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#pragma once

#include "dlnn-mlir/Concepts/Concepts.h"
#include "dlnn-mlir/Dialect/DLNN/Enums.h"
#include "mlir/IR/Builders.h"

//===- Generated includes -------------------------------------------------===//

#include "dlnn-mlir/Dialect/DLNN/Interfaces/ScalarInterface.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::dlnn {

/// Registers external models for ScalarInterface for built-in types.
///
/// Currently supported types:
///     - IntegerType
///     - BFloat16Type
///     - Float16Type
///     - Float32Type
///     - Float64Type
///     - Float80Type
///     - Float128Type
void registerScalarInterfaceModels(MLIRContext &ctx);

/// Scalar type concept.
///
/// Since we can't use a type interface in place of a type constraint in
/// TableGen, we need to use a concept for this.
class ScalarType : public concepts::ConstrainedType<Type, ScalarType> {
public:
    static bool matches(Type type) { return type.isa<ScalarInterface>(); }

    using ConstrainedType::ConstrainedType;

    Value
    createConstant(OpBuilder &builder, Location loc, Attribute value) const
    {
        return cast<ScalarInterface>().createConstant(builder, loc, value);
    }

    Operation* createOp(
        OpBuilder &builder,
        Location loc,
        ScalarOpKind opKind,
        ValueRange operands) const
    {
        return cast<ScalarInterface>().createOp(builder, loc, opKind, operands);
    }
};

/// Scalar value concept.
///
/// See ScalarType for more information.
class Scalar : public concepts::ConstrainedValue<ScalarType> {
public:
    using ConstrainedValue::ConstrainedValue;
};

} // namespace mlir::dlnn
