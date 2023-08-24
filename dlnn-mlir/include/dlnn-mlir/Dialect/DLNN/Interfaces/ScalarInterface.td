//===- ScalarInterface.td - DLNN scalar interface ----------*- tablegen -*-===//
//
// This is the definitions file for the DLNN scalar interface.
//
//===----------------------------------------------------------------------===//

#ifndef DLNN_SCALARINTERFACE
#define DLNN_SCALARINTERFACE

include "dlnn-mlir/Concepts/Concepts.td"
include "mlir/IR/OpBase.td"

def DLNN_ScalarInterface : TypeInterface<"ScalarInterface"> {
    let cppNamespace = "::mlir::dlnn";

    let description = [{
        Interface for a type that can be used as a scalar type in DLNN network
        implementations.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Materializes a constant value of this type from an attribute.
            }],
            /*retTy=*/"::mlir::Value",
            /*methodName=*/"createConstant",
            /*args=*/(ins
                "::mlir::OpBuilder &":$builder,
                "::mlir::Location":$loc,
                "::mlir::Attribute":$value
            ),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{ return ::mlir::Value{}; }]
        >,
        InterfaceMethod<
            /*desc=*/[{
                Materializes an arithmetic operation of the specified kind.
            }],
            /*retTy=*/"::mlir::Operation*",
            /*methodName=*/"createOp",
            /*args=*/(ins
                "::mlir::OpBuilder &":$builder,
                "::mlir::Location":$loc,
                "::mlir::dlnn::ScalarOpKind":$opKind,
                "::mlir::ValueRange":$operands
            ),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{ return nullptr; }]
        >

        // TODO: If we want to be able to do intra-operator transformations, we
        //       will need more methods in this interface to reason about scalar
        //       types. For example:
        //                              _
        //           %0 = splat 0        |
        //           %1 = "add" %0, %A  _|> %1 = %A
        //
        //       Requires knowledge of the neutral element of "add" (0).
    ];
}

def DLNN_ScalarType
        : NativeTypeConcept<"::mlir::dlnn::ScalarType", "scalar">;

#endif
