//===- NodeInterface.td - DLNN node interface --------------*- tablegen -*-===//
//
// This is the definitions file for the DLNN node interface.
//
//===----------------------------------------------------------------------===//

#ifndef DLNN_NODEINTERFACE
#define DLNN_NODEINTERFACE

include "mlir/IR/OpBase.td"

def DLNN_Node : OpInterface<"Node"> {
    let cppNamespace = "::mlir::dlnn";

    let description = [{
        Interface for an operation that represents a node in a DLNN graph.

        A node has a set of inputs and output edges. An edge is a def-use
        dependency between SSA values, i.e. a dataflow dependency in the graph.

        ```mlir
        ..., %in.0, .... = dlnn.node ... // Predecessor
        %out.0, %out.1, ... = dlnn.node %in.0, %in.1, ...
        ... = dlnn.node ..., %out.0, ... // Successor
        ```

        Since values can have multiple uses but only one definition, only input
        edges are always bound to both producer and consumer and therefore
        directly accessible.
    }];

    let methods = [
        InterfaceMethod<
            /*desc=*/[{Gets the range of all input edges.}],
            /*retTy=*/"::mlir::dlnn::EdgeRange",
            /*methodName=*/"getInputs",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{ return $_op->getOperands(); }]
        >,
        InterfaceMethod<
            /*desc=*/[{Gets the range of all output values.}],
            /*retTy=*/"::mlir::ResultRange",
            /*methodName=*/"getOutputs",
            /*args=*/(ins),
            /*methodBody=*/"",
            /*defaultImplementation=*/[{ return $_op->getResults(); }]
        >
    ];

    code extraSharedClassDeclaration = [{
        /// Gets the number of input edges.
        unsigned getNumInputs() { return $_op.getInputs().size(); }

        /// Gets the input edge with the given index.
        ::mlir::dlnn::Edge getInput(unsigned index)
        {
            return $_op.getInputs()[index];
        }

        /// Gets the number of output edges.
        unsigned getNumOutputs() { return $_op.getOutputs().size(); }

        /// Gets the output edge with the given index.
        ::mlir::OpResult getOutput(unsigned index)
        {
            return $_op.getOutputs()[index];
        }
    }];
}

#endif
