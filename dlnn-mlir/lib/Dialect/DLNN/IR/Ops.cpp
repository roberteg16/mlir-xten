/// Implements the DLNN dialect operations.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#include "dlnn-mlir/Dialect/DLNN/IR/Ops.h"

#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::dlnn;

/// Parses a captured SSA operand.
///
/// Format:
///     ssa-id `=` ssa-id `:` type
static ParseResult parseCapture(
    OpAsmParser &p,
    OpAsmParser::UnresolvedOperand &arg,
    OpAsmParser::UnresolvedOperand &src,
    Type &type)
{
    // ssa-id `=` ssa-id `:` type
    if (p.parseOperand(arg)) return failure();
    if (p.parseEqual()) return failure();
    if (p.parseOperand(src)) return failure();
    if (p.parseColon()) return failure();
    if (p.parseType(type)) return failure();

    return success();
}

/// Prints a captured SSA operand.
///
/// See parseCapture() for more details.
static void printCapture(OpAsmPrinter &p, Value arg, Value src)
{
    p << arg << " = " << src << ": " << src.getType();
}

/// Parses a comma-separated list of zero or more captured SSA operands.
///
/// Format:
///     `(` [ capture { `,` capture } ] `)`
static ParseResult parseCaptures(
    OpAsmParser &p,
    SmallVectorImpl<OpAsmParser::Argument> &args,
    SmallVectorImpl<Value> &srcs)
{
    // `(` [ capture { `,` capture } ] `)`
    return p.parseCommaSeparatedList(
        OpAsmParser::Delimiter::Paren,
        [&]() -> ParseResult {
            auto &arg = args.emplace_back();
            OpAsmParser::UnresolvedOperand src;
            if (parseCapture(p, arg.ssaName, src, arg.type)) return failure();
            if (p.resolveOperand(src, arg.type, srcs)) return failure();
            return success();
        });
}

/// Prints a comma-separated list of zero or more captured SSA operands.
///
/// See parseCaptures() for more details.
static void printCaptures(OpAsmPrinter &p, ValueRange args, ValueRange srcs)
{
    auto srcIt = srcs.begin();
    p << '(';
    interleaveComma(args, p, [&](auto arg) {
        assert(srcIt != srcs.end());
        printCapture(p, arg, *srcIt++);
    });
    p << ')';
}

/// Parses a trivial EnclaveOp.
///
/// Format:
///     capture-list [ attr-dict-with-keyword ] region [ `->` type-list ]
static ParseResult parseEnclaveOp(OpAsmParser &p, OperationState &result)
{
    // `(` captures `)`
    SmallVector<OpAsmParser::Argument> args;
    if (parseCaptures(p, args, result.operands)) return failure();

    // [ attr-dict-with-keyword ]
    if (p.parseOptionalAttrDictWithKeyword(result.attributes)) return failure();

    // `{` ... `}`
    if (p.parseRegion(*result.addRegion(), args, true)) return failure();

    // [ `->` type-list ]
    if (succeeded(p.parseOptionalArrow())) {
        if (p.parseTypeList(result.types)) return failure();
    }

    return success();
}

/// Prints a trivial EnclaveOp.
///
/// See parseEnclaveOp() for more details.
static void printEnclaveOp(OpAsmPrinter &p, EnclaveOp op)
{
    p << ' ';
    printCaptures(p, op.getEnclaveBody().getArguments(), op.getCaptures());
    p << ' ';

    p.printOptionalAttrDictWithKeyword(op->getAttrs());
    if (!op->getAttrs().empty()) p << ' ';

    p.printRegion(*op.getEnclaveBody().getParent(), false);

    if (op->getNumResults() > 0) {
        p << " -> ";
        interleaveComma(op->getResultTypes(), p);
    };
}

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "dlnn-mlir/Dialect/DLNN/IR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// NetworkOp
//===----------------------------------------------------------------------===//

ParseResult NetworkOp::parse(OpAsmParser &p, OperationState &result)
{
    // function-op
    return function_interface_impl::parseFunctionOp(
        p,
        result,
        false,
        [](Builder &builder,
           ArrayRef<Type> argTypes,
           ArrayRef<Type> results,
           function_interface_impl::VariadicFlag variadic,
           std::string &) {
            assert(!variadic.isVariadic());
            return builder.getFunctionType(argTypes, results);
        });
}

void NetworkOp::print(OpAsmPrinter &p)
{
    function_interface_impl::printFunctionOp(p, *this, false);
}

//===----------------------------------------------------------------------===//
// EvalOp
//===----------------------------------------------------------------------===//

NetworkOp EvalOp::getNetwork()
{
    SymbolTableCollection lazy;
    return lazy.lookupNearestSymbolFrom<NetworkOp>(*this, getNetworkRefAttr());
}

LogicalResult EvalOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    // There needs to be a reference attribute.
    auto networkRef = getNetworkRefAttr();
    if (!networkRef)
        return emitOpError()
               << "requires a 'network_ref' symbol reference attribute";

    // The reference must point to a Network.
    auto network =
        symbolTable.lookupNearestSymbolFrom<NetworkOp>(*this, networkRef);
    if (!network)
        return emitOpError() << "'" << networkRef.getValue()
                             << "' does not reference a valid network";

    return graph_defaults::verifyUse(
        [&]() { return this->emitOpError(); },
        network,
        this->getOperandTypes(),
        this->getResultTypes());
}

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

ParseResult GraphOp::parse(OpAsmParser &p, OperationState &result)
{
    // function-op
    return function_interface_impl::parseFunctionOp(
        p,
        result,
        false,
        [](Builder &builder,
           ArrayRef<Type> argTypes,
           ArrayRef<Type> results,
           function_interface_impl::VariadicFlag variadic,
           std::string &) {
            assert(!variadic.isVariadic());
            return builder.getFunctionType(argTypes, results);
        });
}

void GraphOp::print(OpAsmPrinter &p)
{
    function_interface_impl::printFunctionOp(p, *this, false);
}

//===----------------------------------------------------------------------===//
// EmbedOp
//===----------------------------------------------------------------------===//

Graph EmbedOp::getGraph()
{
    SymbolTableCollection lazy;
    return lazy.lookupNearestSymbolFrom<Graph>(*this, getGraphRefAttr());
}

LogicalResult EmbedOp::verifySymbolUses(SymbolTableCollection &symbolTable)
{
    // There needs to be a reference attribute.
    auto graphRef = getGraphRefAttr();
    if (!graphRef)
        return emitOpError()
               << "requires a 'graph_ref' symbol reference attribute";

    // The reference must point to a Graph.
    auto graph = symbolTable.lookupNearestSymbolFrom<Graph>(*this, graphRef);
    if (!graph)
        return emitOpError() << "'" << graphRef.getValue()
                             << "' does not reference a valid graph";

    // Recursion is not allowed.
    for (auto parent = (*this)->getParentOfType<GraphOp>(); parent;
         parent = parent->getParentOfType<GraphOp>())
        if (graph == parent)
            return emitOpError() << "recursive embedding detected";

    return graph_defaults::verifyUse(
        [&]() { return this->emitOpError(); },
        graph,
        this->getOperandTypes(),
        this->getResultTypes());
}

//===----------------------------------------------------------------------===//
// SubgraphOp
//===----------------------------------------------------------------------===//

ParseResult SubgraphOp::parse(OpAsmParser &p, OperationState &result)
{
    return parseEnclaveOp(p, result);
}

void SubgraphOp::print(OpAsmPrinter &p) { printEnclaveOp(p, *this); }

//===----------------------------------------------------------------------===//
// NodeOp
//===----------------------------------------------------------------------===//

ParseResult NodeOp::parse(OpAsmParser &p, OperationState &result)
{
    return parseEnclaveOp(p, result);
}

void NodeOp::print(OpAsmPrinter &p) { printEnclaveOp(p, *this); }

//===----------------------------------------------------------------------===//
// ToTensorOp
//===----------------------------------------------------------------------===//

OpFoldResult ToTensorOp::fold(ArrayRef<Attribute>)
{
    if (auto fromTensor = getOrganized().getDefiningOp<FromTensorOp>())
        return fromTensor.getTensor();

    return {};
}

//===----------------------------------------------------------------------===//
// FromTensorOp
//===----------------------------------------------------------------------===//

ParseResult FromTensorOp::parse(OpAsmParser &p, OperationState &result)
{
    // attr-dict
    if (p.parseOptionalAttrDict(result.attributes)) return failure();

    // ssa-id `:` organized-type
    OpAsmParser::UnresolvedOperand tensor;
    if (p.parseOperand(tensor)) return failure();
    if (p.parseColon()) return failure();
    if (p.parseType(result.types.emplace_back())) return failure();

    // Check that the parsed result type is an organized type.
    auto organizedType = result.types.back().dyn_cast<OrganizedType>();
    if (!organizedType)
        return p.emitError(p.getNameLoc())
               << "result type must be an organized type";

    // Resolve the input tensor operand to the associated tensor type.
    if (p.resolveOperand(
            tensor,
            organizedType.getTensorType(),
            result.operands))
        return failure();

    return success();
}

void FromTensorOp::print(OpAsmPrinter &p)
{
    p.printOptionalAttrDict((*this)->getAttrs());
    if (!(*this)->getAttrs().empty()) p << ' ';

    p.printOperand(getTensor());
    p << " : ";
    p.printType(getResult().getType());
}

LogicalResult FromTensorOp::verify()
{
    auto organizedType = getResult().getType().dyn_cast<OrganizedType>();
    if (!organizedType)
        return emitOpError() << "result type must be an organized type";

    if (getOperand().getType() != organizedType.getTensorType())
        return emitOpError() << "operand type (" << getOperand().getType()
                             << ") does not match expected tensor type ("
                             << organizedType.getTensorType() << ")";

    return success();
}

OpFoldResult FromTensorOp::fold(ArrayRef<Attribute>)
{
    if (auto toTensor = getOrganized().getDefiningOp<ToTensorOp>())
        return toTensor.getOrganized();

    return {};
}

//===----------------------------------------------------------------------===//
// DLNNDialect
//===----------------------------------------------------------------------===//

void DLNNDialect::registerOps()
{
    addOperations<
#define GET_OP_LIST
#include "dlnn-mlir/Dialect/DLNN/IR/Ops.cpp.inc"
        >();
}
