//===- Concepts.td - MLIR concepts support -----------------*- tablegen -*-===//
//
// This file provides definitions for MLIR concepts.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONCEPTS
#define MLIR_CONCEPTS

include "mlir/IR/OpBase.td"

class NativeTypeConcept<string concept, string summary>
        : Type<
            /*condition=*/CPred<"$_self.isa<"#concept#">()">,
            /*descr=*/summary,
            /*cppClassName=*/concept
        >;

class NativeAttrConcept<string concept, string summary>
        : Attr<
            /*condition=*/CPred<"$_self.isa<"#concept#">()">,
            /*descr=*/summary
        > {
    let storageType = concept;
    let returnType = "typename "#concept#"::ValueType";
    let convertFromStorage = "$_self.getValue()";
    let valueType = NoneType;
}

#endif
