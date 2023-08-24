//===- Passess.td - DLNN conversion passes -----------------*- tablegen -*-===//
//
// This is the definitions file for the DLNN dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef DLNN_CONVERSION_PASSES
#define DLNN_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertDLNNToFunc : Pass<"convert-dlnn-to-func"> {
    let summary = "Converts DLNN networks and graphs to functions";
    let description = [{
        Converts `dlnn.network` and `dlnn.graph` operations into `func.func`,
        replacing `dlnn.embed` and `dlnn.eval` with `func.call`.

        In order to preserve the symbol uniqueness and properties of the
        `dlnn.graph` ops, the resulting function symbol name will be prefixed
        by the enclosing network name, and the visibility is set to private.
    }];

    let constructor = "mlir::createConvertDLNNToFuncPass()";
    let dependentDialects = [
        "func::FuncDialect"
    ];
}

#endif
