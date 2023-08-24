/// Declares the DLNN volume type interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#pragma once

#include "dlnn-mlir/Dialect/DLNN/Utils/LazyMap.h"
#include "dlnn-mlir/Dialect/DLNN/Utils/Presburger.h"
#include "mlir/IR/Builders.h"

#include <optional>
#include <vector>

namespace mlir::dlnn {

/// Represents a hyperrectangular index domain using OpFoldResult values.
class HyperRectangle {
public:
    /// Range of mutable values.
    using range = MutableArrayRef<OpFoldResult>;
    /// Range of immutable values.
    using const_range = ArrayRef<OpFoldResult>;

    /// Constructs a HyperRectangle from @p offsetsStridesSizes .
    static HyperRectangle fromOffsetStridesSizes(
        OpBuilder &builder,
        Location loc,
        const OffsetsStridesSizes &offsetStridesSizes,
        ArrayRef<Value> symbolValues);

    /// Initializes an undefined HyperRectangle.
    explicit HyperRectangle(unsigned rank) : m_rank(rank), m_values(3 * rank) {}

    /// Gets the number of dimensions.
    unsigned getRank() const { return m_rank; }

    range getOffsets() { return range(m_values).slice(0, getRank()); }
    const_range getOffsets() const
    {
        return const_range(m_values).slice(0, getRank());
    }
    OpFoldResult &getOffset(unsigned dim) { return getOffsets()[dim]; }
    OpFoldResult getOffset(unsigned dim) const { return getOffsets()[dim]; }

    range getStrides() { return range(m_values).slice(getRank(), getRank()); }
    const_range getStrides() const
    {
        return const_range(m_values).slice(getRank(), getRank());
    }
    OpFoldResult &getStride(unsigned dim) { return getStrides()[dim]; }
    OpFoldResult getStride(unsigned dim) const { return getStrides()[dim]; }

    range getSizes() { return range(m_values).slice(2 * getRank(), getRank()); }
    const_range getSizes() const
    {
        return const_range(m_values).slice(2 * getRank(), getRank());
    }
    OpFoldResult &getSize(unsigned dim) { return getSizes()[dim]; }
    OpFoldResult getSize(unsigned dim) const { return getSizes()[dim]; }

    /// Constructs a static shape vector out of the sizes.
    SmallVector<int64_t> makeShape() const;

private:
    unsigned m_rank;
    SmallVector<OpFoldResult> m_values;
};

} // namespace mlir::dlnn

//===- Generated includes -------------------------------------------------===//

#include "dlnn-mlir/Dialect/DLNN/Interfaces/VolumeInterface.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::dlnn {

/// Registers external models for VolumeInterface for built-in types.
///
/// Currently supported types:
///     - RankedTensorType
void registerVolumeInterfaceModels(MLIRContext &ctx);

/// Obtains the index domain for @p volume .
///
/// If @p volume is not a volume, it will be treated as a scalar and have the
/// zero-dimensional universe domain.
std::optional<SetAndSymbolValues>
getIndexDomain(OpBuilder &builder, Value volume);

/// Obtains the type of a subvolume with @p shape .
///
/// If @p volume is not a volume, it will be treated as a scalar, allowing
/// taking its entirety.
Type getSubvolumeType(Type volume, ArrayRef<int64_t> shape);
inline Type getSubvolumeType(Type volume, const HyperRectangle &region)
{
    return getSubvolumeType(volume, region.makeShape());
}

/// Creates an undefined volume of the @p volume type.
///
/// Fails if @p volume is not a volume type.
Value createUndefinedVolume(
    OpBuilder &builder,
    Type volume,
    ArrayRef<OpFoldResult> sizes,
    Location location);

/// Extracts a subvolume @p region from @p volume .
///
/// If @p value is not a volume, it will be treated as a scalar, allowing
/// extracting the whole value only.
Value extractSubvolume(
    OpBuilder &builder,
    Value volume,
    const HyperRectangle &region);

/// Inserts a subvolume @p region into @p volume .
///
/// If @p value is not a volume, it will be treated as a scalar, allowing
/// inserting the whole value only.
Value insertSubvolume(
    OpBuilder &builder,
    Value volume,
    const HyperRectangle &region,
    Value subvolume);

/// Lazy map that caches volume index domain maps.
class VolumeDomainMap : public LazyMap<Value, SetAndSymbolValues> {
public:
    explicit VolumeDomainMap(OpBuilder builder) : m_builder(builder) {}

protected:
    std::optional<SetAndSymbolValues>
    compute(const Value &volume) const override
    {
        // Insert the generated expressions directly after the op that produces
        // the value, if any, otherwise use the current insertion point.
        OpBuilder::InsertionGuard guard(m_builder);
        if (auto op = volume.getDefiningOp())
            m_builder.setInsertionPointAfter(op);
        return getIndexDomain(m_builder, volume);
    }

private:
    mutable OpBuilder m_builder;
};

} // namespace mlir::dlnn
