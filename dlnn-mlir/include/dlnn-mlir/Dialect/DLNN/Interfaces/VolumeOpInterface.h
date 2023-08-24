/// Declares the DLNN volume op interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#pragma once

#include "dlnn-mlir/Dialect/DLNN/Interfaces/VolumeInterface.h"
#include "dlnn-mlir/Dialect/DLNN/Utils/LazyMap.h"
#include "dlnn-mlir/Dialect/DLNN/Utils/OpTiedValue.h"
#include "dlnn-mlir/Dialect/DLNN/Utils/Presburger.h"
#include "mlir/IR/Builders.h"

#include <optional>
#include <vector>

namespace mlir::dlnn {

/// Enumeration of volume access types.
enum class VolumeAccess {
    None = 0,
    Read = 1 << 0,
    Write = 1 << 1,
    LLVM_MARK_AS_BITMASK_ENUM(Write),
    ReadWrite = Read | Write,
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, VolumeAccess access)
{
    if (access == VolumeAccess::None) return os << "None";
    if ((access & VolumeAccess::Read) == VolumeAccess::Read) os << "Read";
    if ((access & VolumeAccess::Write) == VolumeAccess::Write) os << "Write";
    return os;
}

/// Mapping from an iteration space to accessed volume elements.
class VolumeAccessMap {
public:
    /// Initializes a VolumeAccessMap for writing to @p result .
    ///
    /// @pre        `result`
    explicit VolumeAccessMap(OpResult result, Map map)
            : m_volume(result),
              m_access(VolumeAccess::Write),
              m_map(std::move(map))
    {
        assert(result);
    }
    /// Initializes a VolumeAccessMap for @p operand .
    ///
    /// @pre        `operand`
    /// @pre        `access != VolumeAccess::None`
    explicit VolumeAccessMap(OpOperand* operand, VolumeAccess access, Map map)
            : m_volume(operand),
              m_access(access),
              m_map(std::move(map))
    {
        assert(operand);
        assert(access != VolumeAccess::None);
    }
    /// Initializes a VolumeAccessMap for @p volume .
    ///
    /// @pre        `volume`
    /// @pre        `!volume.is<OpResult>() || volume == VolumeAccess::Write`
    explicit VolumeAccessMap(OpTiedValue volume, VolumeAccess access, Map map)
            : m_volume(volume),
              m_access(access),
              m_map(std::move(map))
    {
        assert(volume);
        assert(!volume.is<OpResult>() || access == VolumeAccess::Write);
    }

    /// Gets the VolumeAccess type.
    VolumeAccess getAccess() const { return m_access; }
    /// Gets a value indicating whether this is a read access.
    bool isRead() const
    {
        return (getAccess() & VolumeAccess::Read) == VolumeAccess::Read;
    }
    /// Gets a value indicating whether this is a write access.
    bool isWrite() const
    {
        return (getAccess() & VolumeAccess::Write) == VolumeAccess::Write;
    }

    /// Gets the range volume.
    OpTiedValue getVolume() const { return m_volume; }

    /// Gets the index map.
    const Map &getMap() const { return m_map; }
    /*implicit*/ operator const Map&() const { return getMap(); }

    void dump() const;

private:
    OpTiedValue m_volume;
    VolumeAccess m_access;
    Map m_map;
};

/// Represents the index space of a volume comprehension.
class ComprehensionSpace : public Space {
public:
    /// Initializes a ComprehensionSpace.
    explicit ComprehensionSpace(unsigned rank, unsigned trailing)
            : Space(0, rank + trailing),
              m_rank(rank)
    {}

    /// Gets the number of leading parallel dimensions.
    unsigned getRank() const { return m_rank; }
    /// Gets the number of trailing dimensions.
    unsigned getNumTrailing() const { return getNumDimIds() - getRank(); }

    void dump() const;

private:
    unsigned m_rank;
};

/// Stores a mapping from iteration indices to volume indices.
class VolumeRelation : std::vector<VolumeAccessMap> {
public:
    explicit VolumeRelation(
        OpTiedValue source,
        VolumeAccess access,
        const ComprehensionSpace &domainSpace);

    using vector::value_type;
    using vector::size_type;
    using iterator = vector::const_iterator;

    using vector::size;
    iterator begin() const { return vector::begin(); }
    iterator end() const { return vector::end(); }

    /// Looks up the VolumeAccessMap for @p volume .
    iterator find(OpTiedValue volume) const
    {
        return std::find_if(
            begin(),
            end(),
            [&](auto &map) { return map.getVolume() == volume; });
    }
    /// @copydoc find(OpTiedValue)
    const VolumeAccessMap* lookup(OpTiedValue volume) const
    {
        const auto it = find(volume);
        if (it == end()) return nullptr;
        return &*it;
    }

    /// Gets the iteration domain space.
    const ComprehensionSpace &getDomainSpace() const { return m_domainSpace; }
    /// Gets the VolumeAccessMap of the source volume.
    ///
    /// @pre        `!empty()`
    const VolumeAccessMap &getSourceMap() const { return front(); }
    /// Gets the source volume.
    ///
    /// @pre        `!empty()`
    OpTiedValue getSource() const { return getSourceMap().getVolume(); }
    /// Gets the source operation.
    ///
    /// @pre        `!empty()`
    Operation* getSourceOp() const { return getSource().getOwner(); }

    /// Determines whether @p relationSpace is compatible.
    bool isCompatible(const Space &relationSpace) const
    {
        return getDomainSpace().getNumDimIds()
               == relationSpace.getNumDomainIds();
    }
    /// Determines whether @p map is compatible.
    bool isCompatible(const VolumeAccessMap &map) const
    {
        return isCompatible(map.getMap().getSpace());
    }

    /// Adds @p map to this VolumeRelation.
    ///
    /// @pre        `!lookup(map.getVolume())`
    /// @pre        `isCompatible(map)`
    void unite(VolumeAccessMap map)
    {
        assert(!lookup(map.getVolume()));
        assert(isCompatible(map));
        emplace_back(std::move(map));
    }

    /// Applies @p fn to all contained maps.
    void map(function_ref<void(Map &)> fn)
    {
        for (auto &map : static_cast<vector &>(*this)) {
            Map copy(map);
            fn(copy);
            assert(isCompatible(copy.getSpace()));
            map = VolumeAccessMap(
                map.getVolume(),
                map.getAccess(),
                std::move(copy));
        }
    }

    /// Computes the iteration domain using @p volumeDomains .
    SetAndSymbolValues getIterationDomain(
        VolumeDomainMap &volumeDomains,
        bool allowResults = false) const;
    /// Computes the iteration domain at the source op.
    SetAndSymbolValues getIterationDomain(bool allowResults = false) const
    {
        VolumeDomainMap volumeDomains{OpBuilder(getSourceOp())};
        return getIterationDomain(volumeDomains, allowResults);
    }

    void dump() const;

private:
    VolumeRelation(VolumeAccessMap sourceMap, unsigned syntheticDims)
            : vector(),
              m_domainSpace(
                  sourceMap.getMap().getNumDomainIds() - syntheticDims,
                  syntheticDims)
    {
        // The first map is always the source map.
        vector::emplace_back(std::move(sourceMap));
    }

    ComprehensionSpace m_domainSpace;
};

} // namespace mlir::dlnn

//===- Generated includes -------------------------------------------------===//

#include "dlnn-mlir/Dialect/DLNN/Interfaces/VolumeOpInterface.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::dlnn {

/// Tries to obtain the VolumeRelation for @p source .
///
/// @pre        `source`
std::optional<VolumeRelation> getVolumeRelation(OpTiedValue source);

/// Lazy map that caches VolumeRelation maps.
class VolumeRelationMap : public LazyMap<OpTiedValue, VolumeRelation> {
protected:
    std::optional<VolumeRelation>
    compute(const OpTiedValue &source) const override
    {
        return getVolumeRelation(source);
    }
};

} // namespace mlir::dlnn
