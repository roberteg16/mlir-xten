/// Declaration of the DLNN dialect types.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#pragma once

#include "dlnn-mlir/Dialect/DLNN/IR/Base.h"
#include "dlnn-mlir/Dialect/DLNN/Utils/Organization.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/TypeSwitch.h"

//===- Generated includes -------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "dlnn-mlir/Dialect/DLNN/IR/Types.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::dlnn {

/// Organized type concept.
///
/// An organized type is a multi-dimensional container type that assigns
/// semantics to each of its dimensions.
///
/// Since inheritance doesn't work well for types, we use a concept to represent
/// the common functionality between the implicit tensor organization types
/// FeatureMapType and WeightsType.
class OrganizedType : public ConstrainedType<Type, OrganizedType> {
public:
    static bool matches(FeatureMapType) { return true; }
    static bool matches(WeightsType) { return true; }
    static bool matches(Type type)
    {
        return type.isa<FeatureMapType, WeightsType>();
    }

    using ConstrainedType::ConstrainedType;

    /*implicit*/ OrganizedType(FeatureMapType fmTy)
            : OrganizedType(getImpl(fmTy))
    {}
    /*implicit*/ OrganizedType(WeightsType wgtTy)
            : OrganizedType(getImpl(wgtTy))
    {}

    /// Gets the element scalar type.
    ScalarType getScalarType() const
    {
        return llvm::TypeSwitch<OrganizedType, ScalarType>(*this)
            .Case<FeatureMapType, WeightsType>(
                [](auto type) { return type.getScalarType(); });
    }
    /// Gets the dimensions with size semantics.
    ShapeRef getSizes() const
    {
        return llvm::TypeSwitch<OrganizedType, ShapeRef>(*this)
            .Case<FeatureMapType, WeightsType>(
                [](auto type) { return type.getSizes(); });
    }
    /// Gets the organization map.
    AffineMap getOrganization() const
    {
        return llvm::TypeSwitch<OrganizedType, AffineMap>(*this)
            .Case<FeatureMapType, WeightsType>(
                [](auto type) { return type.getOrganization(); });
    }
    /// Gets the lowered tensor type.
    RankedTensorType getTensorType() const
    {
        return llvm::TypeSwitch<OrganizedType, RankedTensorType>(*this)
            .Case<FeatureMapType, WeightsType>(
                [](auto type) { return type.getTensorType(); });
    }
};

/// Organized value concept.
///
/// See OrganizedType for more information.
class Organized : public ConstrainedValue<OrganizedType> {
public:
    using ConstrainedValue::ConstrainedValue;

    /// @copydoc OrganizedType::getScalarType()
    ScalarType getScalarType() const { return getType().getScalarType(); }
    /// @copydoc OrganizedType::getSizes()
    ShapeRef getSizes() const { return getType().getSizes(); }
    /// @copydoc OrganizedType::getOrganization()
    AffineMap getOrganization() const { return getType().getOrganization(); }
    /// @copydoc OrganizedType::getTensorType()
    RankedTensorType getTensorType() const { return getType().getTensorType(); }
};

} // namespace mlir::dlnn
