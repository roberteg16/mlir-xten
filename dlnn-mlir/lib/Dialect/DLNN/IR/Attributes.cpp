/// Implements the DLNN dialect attributes.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@amd.com)

#include "dlnn-mlir/Dialect/DLNN/IR/Attributes.h"

#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "dlnn-attributes"

using namespace mlir;
using namespace mlir::dlnn;

//===- Generated implementation -------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "dlnn-mlir/Dialect/DLNN/IR/Attributes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DLNNDialect
//===----------------------------------------------------------------------===//

// void DLNNDialect::registerAttributes()
// {
//     addAttributes<
// #define GET_ATTRDEF_LIST
// #include "dlnn-mlir/Dialect/DLNN/IR/Attributes.cpp.inc"
//         >();
// }

// Attribute DLNNDialect::parseAttribute(DialectAsmParser& parser, Type type)
// const
// {
//     StringRef attrTag;
//     if (failed(parser.parseKeyword(&attrTag))) return Attribute();

//     Attribute genAttr;
//     auto parseResult = generatedAttributeParser(parser, attrTag, type,
//     genAttr); if (parseResult.hasValue()) return genAttr;

//     parser.emitError(parser.getNameLoc(), "unknown dlnn attribute: ")
//         << attrTag;
//     return Attribute();
// }

// void DLNNDialect::printAttribute(Attribute attr, DialectAsmPrinter& printer)
//     const
// {
//     if (failed(generatedAttributePrinter(attr, printer)))
//         llvm_unreachable("unexpected 'dlnn' attribute kind");
// }
