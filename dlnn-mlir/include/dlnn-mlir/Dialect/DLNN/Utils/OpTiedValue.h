/// Declares the OpTiedValue utility type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::dlnn {

/// Reference to a value that is strongly tied to an op (an operand or result).
///
/// Instead of storing a pair<Operation*, Value>, this PointerUnion can be used
/// to wrap either an operand or a result of an operation, allowing access to
/// both the op and the value.
class OpTiedValue : public llvm::PointerUnion<OpResult, OpOperand*> {
public:
    using PointerUnion::PointerUnion;
    /*implicit*/ OpTiedValue(PointerUnion base) : PointerUnion(base) {}

    /// Gets the operation that this value is tied to.
    Operation* getOwner() const
    {
        if (auto result = dyn_cast<OpResult>()) return result.getOwner();
        return get<OpOperand*>()->getOwner();
    }
    /// @copydoc getOwner()
    /*implicit*/ operator Operation*() const { return getOwner(); }

    /// Gets the underlying value.
    Value getValue() const
    {
        if (auto result = dyn_cast<OpResult>()) return result;
        return get<OpOperand*>()->get();
    }
    /// @copydoc getValue()
    /*implicit*/ operator Value() const { return getValue(); }

    /// Gets the underlying type.
    Type getType() const { return getValue().getType(); }

    /// Writes @p value to @p out .
    friend llvm::raw_ostream &
    operator<<(llvm::raw_ostream &out, OpTiedValue value)
    {
        if (auto res = value.dyn_cast<OpResult>())
            out << "result #" << res.getResultNumber();
        else
            out << "operand #" << value.get<OpOperand*>()->getOperandNumber();
        out << " of ";
        value.getOwner()->print(out);
        return out;
    }
};

} // namespace mlir::dlnn

// Enable usage of OpTiedValue in llvm::DenseMap.
template<>
struct llvm::DenseMapInfo<mlir::dlnn::OpTiedValue> {
    using Self = mlir::dlnn::OpTiedValue;
    using Base = llvm::DenseMapInfo<
        llvm::PointerUnion<mlir::OpResult, mlir::OpOperand*>>;

    static Self getEmptyKey() { return Self(Base::getEmptyKey()); }
    static Self getTombstoneKey() { return Self(Base::getTombstoneKey()); }
    static unsigned getHashValue(const Self &self)
    {
        return Base::getHashValue(self);
    }
    static bool isEqual(const Self &lhs, const Self &rhs) { return lhs == rhs; }
};
