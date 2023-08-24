//===- Base.td - DLNN dialect base -------------------------*- tablegen -*-===//
//
// This is the definitions file for the DLNN dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef DLNN_BASE
#define DLNN_BASE

include "dlnn-mlir/Concepts/Concepts.td"
include "dlnn-mlir/Dialect/DLNN/Enums.td"
include "dlnn-mlir/Dialect/DLNN/Interfaces/EnclaveOpInterface.td"
include "dlnn-mlir/Dialect/DLNN/Interfaces/GraphInterface.td"
include "dlnn-mlir/Dialect/DLNN/Interfaces/NodeInterface.td"
include "dlnn-mlir/Dialect/DLNN/Interfaces/ScalarInterface.td"
include "dlnn-mlir/Dialect/DLNN/Interfaces/VolumeInterface.td"
include "dlnn-mlir/Dialect/DLNN/Interfaces/VolumeOpInterface.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def DLNN_Dialect : Dialect {
    let name = "dlnn";
    let cppNamespace = "::mlir::dlnn";
    let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;

    let summary = "The Deep Learning Neural Network dialect";
    let description = [{
        TODO
    }];

    let dependentDialects = [
        "arith::ArithmeticDialect",
        "math::MathDialect",
        "tensor::TensorDialect"
    ];

    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();

        ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
        void printType(
            ::mlir::Type type,
            ::mlir::DialectAsmPrinter &printer
        ) const override;
    }];

    let hasConstantMaterializer = 1;
    let useDefaultTypePrinterParser = 0;
}

// Template for attributes.
//class DLNN_Attr<string name, list<Trait> traits = []>
//      : AttrDef<DLNN_Dialect, name, traits>;
// Template for ops.
class DLNN_Op<string mnemonic, list<Trait> traits = []>
        : Op<DLNN_Dialect, mnemonic, traits>;
// Template for types.
class DLNN_Type<
        string name,
        list<Trait> traits = [],
        string baseClass = "::mlir::Type">
        : TypeDef<DLNN_Dialect, name, traits, baseClass>;

#endif
