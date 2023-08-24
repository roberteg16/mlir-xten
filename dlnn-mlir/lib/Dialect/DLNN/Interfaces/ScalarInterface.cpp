/// Implements the DLNN dialect scalar interface.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#include "dlnn-mlir/Dialect/DLNN/Interfaces/ScalarInterface.h"

#include "dlnn-mlir/Dialect/DLNN/IR/DLNN.h"

using namespace mlir;
using namespace mlir::dlnn;

//===- Generated implementation -------------------------------------------===//

#include "dlnn-mlir/Dialect/DLNN/Interfaces/ScalarInterface.cpp.inc"

//===----------------------------------------------------------------------===//

namespace {

/// CRTP base for implementing an external model for a builtin type.
template<class Derived, class T>
struct BuiltinModelBase : ScalarInterface::ExternalModel<Derived, T> {
    Value
    createConstant(Type self, OpBuilder &builder, Location loc, Attribute value)
        const
    {
        // Bail if called for an attribute of a different type.
        if (!value || value.getType() != self) return Value{};

        // Always materialize as an `arith.constant`.
        return builder.create<arith::ConstantOp>(loc, value).getResult();
    }

    Operation* createOp(
        Type self,
        OpBuilder &builder,
        Location loc,
        ScalarOpKind kind,
        ValueRange operands) const
    {
        // Bail if called for an operand of a different type.
        if (llvm::any_of(operands, [&](auto op) {
                return op.getType() != self;
            }))
            return nullptr;

        // Delegate to concrete type implementation.
        return static_cast<const Derived &>(*this)
            .createOpImpl(self.cast<T>(), builder, loc, kind, operands);
    }
};

/// External model for IntegerType.
struct IntModel : BuiltinModelBase<IntModel, IntegerType> {
    Operation* createOpImpl(
        IntegerType self,
        OpBuilder &builder,
        Location loc,
        ScalarOpKind kind,
        ValueRange operands) const
    {
        switch (kind) {
        case ScalarOpKind::Add:
            return builder.create<arith::AddIOp>(loc, operands);
        case ScalarOpKind::And:
            return builder.create<arith::AndIOp>(loc, operands);
        case ScalarOpKind::Div:
            if (self.isSigned())
                return builder.create<arith::DivSIOp>(loc, operands);
            if (self.isUnsigned())
                return builder.create<arith::DivUIOp>(loc, operands);

            // Unsupported if signedness is not known.
            return nullptr;
        case ScalarOpKind::Max:
            if (self.isSigned())
                return builder.create<arith::MaxSIOp>(loc, operands);
            if (self.isUnsigned())
                return builder.create<arith::MaxUIOp>(loc, operands);

            // Unsupported if signedness is not known.
            return nullptr;
        case ScalarOpKind::Min:
            if (self.isSigned())
                return builder.create<arith::MinSIOp>(loc, operands);
            if (self.isUnsigned())
                return builder.create<arith::MinUIOp>(loc, operands);

            // Unsupported if signedness is not known.
            return nullptr;
        case ScalarOpKind::Mod:
            if (self.isSigned())
                return builder.create<arith::RemSIOp>(loc, operands);
            if (self.isUnsigned())
                return builder.create<arith::RemUIOp>(loc, operands);

            // Unsupported if signedness is not known.
            return nullptr;
        case ScalarOpKind::Mul:
            return builder.create<arith::MulIOp>(loc, operands);
        case ScalarOpKind::Not:
            // ~a = a ^ ty(-1)
            return builder.create<arith::XOrIOp>(
                loc,
                operands.front(),
                builder.createOrFold<arith::ConstantOp>(
                    loc,
                    IntegerAttr::get(
                        self,
                        APInt::getAllOnes(self.getWidth()))));
        case ScalarOpKind::Or:
            return builder.create<arith::OrIOp>(loc, operands);
        case ScalarOpKind::Sub:
            return builder.create<arith::SubIOp>(loc, operands);
        case ScalarOpKind::Xor:
            return builder.create<arith::XOrIOp>(loc, operands);

        default:
            // Unsupported operation.
            return nullptr;
        }
    }
};

/// External model for a concrete FloatType.
template<class Float>
struct FloatModel : BuiltinModelBase<FloatModel<Float>, Float> {
    Operation* createOpImpl(
        Float self,
        OpBuilder &builder,
        Location loc,
        ScalarOpKind kind,
        ValueRange operands) const
    {
        switch (kind) {
        case ScalarOpKind::Abs:
            return builder.create<math::AbsOp>(loc, operands);
        case ScalarOpKind::Add:
            return builder.create<arith::AddFOp>(loc, operands);
        case ScalarOpKind::Atan:
            return builder.create<math::AtanOp>(loc, operands);
        case ScalarOpKind::Ceil:
            return builder.create<math::CeilOp>(loc, operands);
        case ScalarOpKind::Cos:
            return builder.create<math::CosOp>(loc, operands);
        case ScalarOpKind::Div:
            return builder.create<arith::DivFOp>(loc, operands);
        case ScalarOpKind::Erf:
            return builder.create<math::ErfOp>(loc, operands);
        case ScalarOpKind::Exp:
            return builder.create<math::ExpOp>(loc, operands);
        case ScalarOpKind::Floor:
            return builder.create<math::FloorOp>(loc, operands);
        case ScalarOpKind::Log:
            return builder.create<math::LogOp>(loc, operands);
        case ScalarOpKind::Max:
            return builder.create<arith::MaxFOp>(loc, operands);
        case ScalarOpKind::Min:
            return builder.create<arith::MinFOp>(loc, operands);
        case ScalarOpKind::Mod:
            return builder.create<arith::RemFOp>(loc, operands);
        case ScalarOpKind::Mul:
            return builder.create<arith::MulFOp>(loc, operands);
        case ScalarOpKind::Neg:
            return builder.create<arith::NegFOp>(loc, operands);
        case ScalarOpKind::Pow:
            return builder.create<math::PowFOp>(loc, operands);
        case ScalarOpKind::Sin:
            return builder.create<math::SinOp>(loc, operands);
        case ScalarOpKind::Sub:
            return builder.create<arith::SubFOp>(loc, operands);
        case ScalarOpKind::Tanh:
            return builder.create<math::TanhOp>(loc, operands);

        // Unsupported operation.
        default: return nullptr;
        }
    }
};

} // namespace

void mlir::dlnn::registerScalarInterfaceModels(MLIRContext &ctx)
{
    IntegerType::attachInterface<IntModel>(ctx);

    BFloat16Type::attachInterface<FloatModel<BFloat16Type>>(ctx);
    Float16Type::attachInterface<FloatModel<Float16Type>>(ctx);
    Float32Type::attachInterface<FloatModel<Float32Type>>(ctx);
    Float64Type::attachInterface<FloatModel<Float64Type>>(ctx);
    Float80Type::attachInterface<FloatModel<Float80Type>>(ctx);
    Float128Type::attachInterface<FloatModel<Float128Type>>(ctx);
}
