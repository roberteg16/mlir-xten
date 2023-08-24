/// Implements the DLNN dialect volume interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#include "dlnn-mlir/Dialect/DLNN/Interfaces/VolumeInterface.h"

#include "dlnn-mlir/Dialect/DLNN/IR/DLNN.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::dlnn;
using namespace mlir::presburger;

static OpFoldResult getConstantIndexOrValue(Value value)
{
    if (auto op = value.getDefiningOp<arith::ConstantIndexOp>())
        return op.getValue();

    return value;
}

static OpFoldResult buildSymbolExpr(
    OpBuilder &builder,
    Location loc,
    ArrayRef<int64_t> coeffs,
    ArrayRef<Value> symbolValues)
{
    assert(
        symbolValues.size() == coeffs.size() - 1
        && "mismatched number of symbols");

    // Turn the coefficients into an AffineExpr.
    auto expr = getAffineExprFromFlatForm(
        coeffs,
        0,
        symbolValues.size(),
        {},
        builder.getContext());
    // Turn the expression into a single-result AffineMap.
    auto map = AffineMap::get(0, symbolValues.size(), expr);
    // Evaluate the map.
    return getConstantIndexOrValue(
        builder.createOrFold<AffineApplyOp>(loc, map, symbolValues));
}

HyperRectangle HyperRectangle::fromOffsetStridesSizes(
    OpBuilder &builder,
    Location loc,
    const OffsetsStridesSizes &offsetStridesSizes,
    ArrayRef<Value> symbolValues)
{
    assert(
        offsetStridesSizes.getDomainSpace().getNumSymbolIds()
            == symbolValues.size()
        && "mismatched number of symbols");

    // Create an undefined result.
    HyperRectangle result(offsetStridesSizes.getRank());

    // Create and assign all the expressions.
    const auto build = [&](ArrayRef<int64_t> coeffs) {
        return buildSymbolExpr(builder, loc, coeffs, symbolValues);
    };
    for (auto dim : iota_range<unsigned>(0, result.getRank(), false)) {
        result.getOffset(dim) = build(offsetStridesSizes.getOffset(dim));
        result.getStride(dim) = build(offsetStridesSizes.getStride(dim));
        result.getSize(dim) = build(offsetStridesSizes.getSize(dim));
    }

    return result;
}

SmallVector<int64_t> HyperRectangle::makeShape() const
{
    return to_vector(map_range(
        getSizes(),
        [](OpFoldResult x) {
            if (auto attr =
                    x.dyn_cast<Attribute>().dyn_cast_or_null<IntegerAttr>())
                return attr.getValue().getSExtValue();
            return ShapedType::kDynamicSize;
        }));
}

//===- Generated implementation -------------------------------------------===//

#include "dlnn-mlir/Dialect/DLNN/Interfaces/VolumeInterface.cpp.inc"

//===----------------------------------------------------------------------===//

namespace {

/// CRTP base for implementing an external model for a builtin type.
template<class Derived, class T>
struct BuiltinModelBase : VolumeInterface::ExternalModel<Derived, T> {
    std::optional<SetAndSymbolValues>
    getIndexDomain(Type self, OpBuilder &builder, Value value) const
    {
        if (!value || value.getType() != self) return std::nullopt;

        return static_cast<const Derived &>(*this).getIndexDomainImpl(
            self.cast<T>(),
            builder,
            value);
    }

    Type getSubvolumeType(Type self, ArrayRef<int64_t> shape) const
    {
        return static_cast<const Derived &>(*this).getSubvolumeTypeImpl(
            self.cast<T>(),
            shape);
    }

    Value createUndefinedVolume(
        Type self,
        OpBuilder &builder,
        ArrayRef<OpFoldResult> sizes,
        Location location) const
    {
        return static_cast<const Derived &>(*this).createUndefinedVolumeImpl(
            self.cast<T>(),
            builder,
            sizes,
            location);
    }

    Value extractSubvolume(
        Type self,
        OpBuilder &builder,
        Value source,
        const HyperRectangle &region) const
    {
        if (!source || source.getType() != self) return Value{};

        return static_cast<const Derived &>(*this)
            .extractSubvolumeImpl(self.cast<T>(), builder, source, region);
    }

    Value insertSubvolume(
        Type self,
        OpBuilder &builder,
        Value source,
        const HyperRectangle &region,
        Value subvolume) const
    {
        if (!source || source.getType() != self) return Value{};

        return static_cast<const Derived &>(*this).insertSubvolumeImpl(
            self.cast<T>(),
            builder,
            source,
            region,
            subvolume);
    }
};

/// Obtains an index domain for a (dynamic) shape.
///
/// For every dynamic dimension, the index domain will obtain an upper bound
/// based on a newly inserted symbol value. These symbols are created in order
/// of the dynamic dimensions appearing in @p shape .
static Set getIndexDomain(ShapeRef shape)
{
    const auto rank = shape.size();
    const auto numDynamicDims = count(shape, ShapedType::kDynamicSize);
    const auto space = Space::getSetSpace(rank, numDynamicDims);
    Set result(rank * 2, 0, space.getNumIds() + 1, space);

    SmallVector<int64_t> coeff(result.getNumCols(), 0);
    unsigned dynamicDim = result.getIdKindOffset(IdKind::Symbol);
    for (auto [dim, sz] : enumerate(shape)) {
        coeff[dim] = 1;
        result.addInequality(coeff);
        coeff[dim] = -1;
        if (sz != ShapedType::kDynamicSize) {
            coeff.back() = sz - 1;
            result.addInequality(coeff);
            coeff.back() = 0;
        } else {
            coeff.back() = -1;
            coeff[dynamicDim] = 1;
            result.addInequality(coeff);
            coeff.back() = coeff[dynamicDim++] = 0;
        }
        coeff[dim] = 0;
    }

    return result;
}

/// External model for RankedTensorType.
struct RankedTensorModel
        : BuiltinModelBase<RankedTensorModel, RankedTensorType> {
    std::optional<SetAndSymbolValues> getIndexDomainImpl(
        RankedTensorType self,
        OpBuilder &builder,
        Value value) const
    {
        // Initialize the result with the (symbolic) index domain.
        SetAndSymbolValues result(::getIndexDomain(self.getShape()));

        // For every dynamic dimension, obtain the runtime extent and store
        // that value in the symbol-to-value mapping of the result.
        unsigned dynamicDim = 0;
        for (auto [dim, sz] : enumerate(self.getShape())) {
            if (sz != ShapedType::kDynamicSize) continue;
            result.setSymbolValue(
                dynamicDim++,
                builder.create<tensor::DimOp>(value.getLoc(), value, dim));
        }

        return result;
    }

    Type
    getSubvolumeTypeImpl(RankedTensorType self, ArrayRef<int64_t> shape) const
    {
        return self.cloneWith(shape, self.getElementType());
    }

    Value createUndefinedVolumeImpl(
        RankedTensorType self,
        OpBuilder &builder,
        ArrayRef<OpFoldResult> sizes,
        Location location) const
    {
        return builder.create<linalg::InitTensorOp>(
            location,
            sizes,
            self.getElementType());
    }

    Value extractSubvolumeImpl(
        RankedTensorType self,
        OpBuilder &builder,
        Value source,
        const HyperRectangle &region) const
    {
        return builder
            .create<tensor::ExtractSliceOp>(
                source.getLoc(),
                source,
                region.getOffsets(),
                region.getSizes(),
                region.getStrides())
            .getResult();
    }

    Value insertSubvolumeImpl(
        RankedTensorType self,
        OpBuilder &builder,
        Value source,
        const HyperRectangle &region,
        Value subvolume) const
    {
        return builder
            .create<tensor::InsertSliceOp>(
                source.getLoc(),
                subvolume,
                source,
                region.getOffsets(),
                region.getSizes(),
                region.getStrides())
            .getResult();
    }
};

} // namespace

void mlir::dlnn::registerVolumeInterfaceModels(MLIRContext &ctx)
{
    RankedTensorType::attachInterface<RankedTensorModel>(ctx);
}

std::optional<SetAndSymbolValues>
mlir::dlnn::getIndexDomain(OpBuilder &builder, Value volume)
{
    assert(volume);

    if (auto iface = volume.getType().dyn_cast<VolumeInterface>())
        return iface.getIndexDomain(builder, volume);

    // Scalar case.
    return SetAndSymbolValues(Set(0, 0, 1, Space::getSetSpace()));
}

Type mlir::dlnn::getSubvolumeType(Type volume, ArrayRef<int64_t> shape)
{
    assert(volume);

    if (auto iface = volume.dyn_cast<VolumeInterface>())
        return iface.getSubvolumeType(shape);

    // Scalar case.
    if (shape.size() == 0) return volume;

    return {};
}

Value mlir::dlnn::createUndefinedVolume(
    OpBuilder &builder,
    Type volume,
    ArrayRef<OpFoldResult> sizes,
    Location location)
{
    assert(volume);

    if (auto iface = volume.dyn_cast<VolumeInterface>())
        return iface.createUndefinedVolume(builder, sizes, location);

    return {};
}

Value mlir::dlnn::extractSubvolume(
    OpBuilder &builder,
    Value volume,
    const HyperRectangle &region)
{
    assert(volume);

    if (auto iface = volume.getType().dyn_cast<VolumeInterface>())
        return iface.extractSubvolume(builder, volume, region);

    // Scalar case.
    if (region.getRank() == 0) return volume;

    return {};
}

Value mlir::dlnn::insertSubvolume(
    OpBuilder &builder,
    Value volume,
    const HyperRectangle &region,
    Value subvolume)
{
    assert(volume);

    if (auto iface = volume.getType().dyn_cast<VolumeInterface>())
        return iface.insertSubvolume(builder, volume, region, subvolume);

    // Scalar case.
    if (region.getRank() == 0) return subvolume;

    return {};
}
