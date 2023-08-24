/// Implements the DLNN dialect base.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#include "dlnn-mlir/Dialect/DLNN/IR/Base.h"

#include "dlnn-mlir/Dialect/DLNN/IR/DLNN.h"
#include "dlnn-mlir/Dialect/DLNN/Utils/STLExtras.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::dlnn;

//===- Generated implementation -------------------------------------------===//

#include "dlnn-mlir/Dialect/DLNN/IR/Base.cpp.inc"

//===----------------------------------------------------------------------===//

/// Implements the DialectInlinerInterface for the DLNN dialect.
struct DLNNInlinerInterface : DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    virtual bool
    isLegalToInline(Operation* call, Operation*, bool) const override
    {
        // Eval ops can never be inlined.
        return !isa<EvalOp>(call);
    }

    virtual bool isLegalToInline(Region*, Region*, bool, BlockAndValueMapping &)
        const override
    {
        // All regions are legal to inline.
        return true;
    }

    virtual bool
    isLegalToInline(Operation*, Region* dest, bool, BlockAndValueMapping &)
        const override
    {
        // We can only inline into our container ops.
        return isa<NetworkOp, GraphOp, SubgraphOp>(dest->getParentOp());
    }

    virtual void handleTerminator(
        Operation* op,
        ArrayRef<Value> valuesToReplace) const override
    {
        // Replace the values with the operands to the OutputOp.
        auto output = cast<OutputOp>(op);
        assert(valuesToReplace.size() == output.getNumOperands());

        for (auto [idx, value] : enumerate(valuesToReplace))
            value.replaceAllUsesWith(output.getOperand(idx));
    }
};

//===----------------------------------------------------------------------===//
// DLNNDialect
//===----------------------------------------------------------------------===//

Operation* DLNNDialect::materializeConstant(
    OpBuilder &builder,
    Attribute value,
    Type type,
    Location loc)
{
    // TODO: Implement.
    return nullptr;
}

void DLNNDialect::initialize()
{
    // Delegate to the registry methods.
    registerOps();
    registerTypes();

    registerScalarInterfaceModels(*getContext());
    registerVolumeInterfaceModels(*getContext());

    addInterfaces<DLNNInlinerInterface>();
}
