//===- Ops.td - DLNN op declarations -----------------------*- tablegen -*-===//
//
// This is the definitions file for the DLNN dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef DLNN_OPS
#define DLNN_OPS

include "dlnn-mlir/Dialect/DLNN/IR/Attributes.td"

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// NetworkOp
//===----------------------------------------------------------------------===//

def DLNN_NetworkOp : DLNN_Op<"network", [
            SingleBlockImplicitTerminator<"OutputOp">,
            HasOnlyGraphRegion,
            RegionKindInterface,
            Symbol,
            SymbolTable,
            IsolatedFromAbove,
            FunctionOpInterface,
            CallableOpInterface,
            DLNN_Graph,
            OpAsmOpInterface]> {
    let summary = "Defines the entry point into a DLNN network";
    let description = [{
        The `dlnn.network` operation is a container for `dlnn.graph` operations,
        which is itself a graph.

        A network can be invoked from outside the DLNN dialect via the
        `dlnn.eval` operation.

        Example:
        ```mlir
        dlnn.network @my_network(%arg0: !fm<3xf16[108,108]>) -> !fm<128xf16> {
            %0 = dlnn.subgraph (%c0 = %arg0 : !fm<3xf16[108,108]>) {
                // Implementation...
            } -> !fm<128xf16>
            dlnn.graph @my_graph(%arg0: !fm<128xf16>) -> !fm<128xf16> {
                // Implementation...
            }
            %1 = dlnn.embed @my_graph(%0) : (!fm<128xf16>) -> (!fm<128xf16>)
            dlnn.output %1 : !fm<128xf16>
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type);
    let regions = (region SizedRegion<1>:$content);

    let hasCustomAssemblyFormat = 1;

    code extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface
    //===------------------------------------------------------------------===//

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    Region *getCallableRegion() { return &getContent(); }
    ArrayRef<Type> getCallableResults()
    {
        return getFunctionType().getResults();
    }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    static StringRef getDefaultDialect() { return "dlnn"; }
    }];
}

//===----------------------------------------------------------------------===//
// EvalOp
//===----------------------------------------------------------------------===//

def DLNN_EvalOp : DLNN_Op<"eval", [
            CallOpInterface,
            DeclareOpInterfaceMethods<
                SymbolUserOpInterface,
                ["verifySymbolUses"]>]> {
    let summary = "Evaluates a DLNN network once";
    let description = [{
        The `dlnn.eval` operation is used to provide inputs to, evaluate and
        collect outputs from a `dlnn.network`.

        This op is similar to `func.func` as it essentially implements a call
        to a network, a single instance of which is assumed to have been
        implemented elsewhere.

        Example:
        ```mlir
        dlnn.network @my_network(%arg0: !fm<3xf16[108,108]>) -> !fm<128xf16> {
            // Implementation ...
        }
        func.func @my_func(%arg0 : tensor<3x108x108xf32>) -> tensor<128xf16> {
            %0 = dlnn.from_tensor %arg0 : !fm<3xf16[108,108]>
            %1 = dlnn.eval @my_network(%0) : (!fm<3xf16[108,108]>) -> (!fm<128xf16>)
            %2 = dlnn.to_tensor %1 : !fm<128xf16>
            return %2 : tensor<128xf16>
        }
        ```
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$network_ref,
        Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>:$results);

    let assemblyFormat = [{
        $network_ref `(` $operands `)` attr-dict
        `:` functional-type($operands, results)
    }];

    code extraClassDeclaration = [{
    /// Gets the Network that this op references.
    NetworkOp getNetwork();

    //===------------------------------------------------------------------===//
    // CallOpInterface
    //===------------------------------------------------------------------===//

    CallInterfaceCallable getCallableForCallee() { return getNetworkRefAttr(); }
    operand_range getArgOperands() { return getOperands(); }
    }];
}

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

def DLNN_GraphOp : DLNN_Op<"graph", [
            HasParent<"NetworkOp">,
            SingleBlockImplicitTerminator<"OutputOp">,
            HasOnlyGraphRegion,
            RegionKindInterface,
            Symbol,
            IsolatedFromAbove,
            FunctionOpInterface,
            CallableOpInterface,
            DLNN_Graph,
            OpAsmOpInterface]> {
    let summary = "Defines an operator graph";
    let description = [{
        The `dlnn.graph` operation defines a named graph of DLNN operators. It
        is similar to a `func.func` definition, except that:

            - Its body is a graph region, meaning that the order of execution
              of its contents is not specified. All scheduling constraints must
              be imposed through def-use dataflow on SSA values.

            - An invocation of a graph (i.e. a `func.call` in `func` world) has
              embedding instead of call semantics. Conceptually, every embedding
              of a graph has its own copy of associated execution resources.

        The latter property enforces graphs to be reentrant regions of code.

        Example:
        ```mlir
        dlnn.graph @my_graph(%arg0: !fm<3xf16[108,108]>) -> !fm<128xf16> {
            // Implementation...
            dlnn.output %result : !fm<128xf16>
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type);
    let regions = (region SizedRegion<1>:$content);

    let hasCustomAssemblyFormat = 1;

    code extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface
    //===------------------------------------------------------------------===//

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    Region *getCallableRegion() { return &getContent(); }
    ArrayRef<Type> getCallableResults()
    {
        return getFunctionType().getResults();
    }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface
    //===------------------------------------------------------------------===//

    static StringRef getDefaultDialect() { return "dlnn"; }
    }];
}

//===----------------------------------------------------------------------===//
// EmbedOp
//===----------------------------------------------------------------------===//

def DLNN_EmbedOp : DLNN_Op<"embed", [
            HasParent<"NetworkOp, GraphOp, SubgraphOp">,
            IsolatedFromAbove,
            CallOpInterface,
            DeclareOpInterfaceMethods<
                SymbolUserOpInterface,
                ["verifySymbolUses"]>,
            DLNN_Node,
            DLNN_Graph]> {
    let summary = "Embeds a named operator subgraph";
    let description = [{
        The `dlnn.embed` operation performs a logical embedding of a named
        `dlnn.graph` as a subgraph within the current graph.

        This is different from a plain function call, since all nodes and their
        associated execution resources are logically duplicated and thus unique
        for every embedding-site. As a result, recursion is strictly prohibited.

        Example:
        ```mlir
        dlnn.graph @graph1(%arg0: !fm<3xf16[108,108]>) -> !fm<128xf16> {
            // Implementation...
            dlnn.output %result : !fm<128xf16>
        }
        dlnn.graph @graph2(%arg0: !fm<3xf16[108,108]>) -> !fm<128xf16> {
            %result = dlnn.embed @graph1(%arg0)
                    : (!fm<3xf16[108,108]>) -> (!fm<128xf16>)
            dlnn.output %result : !fm<128xf16>
        }
        ```
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$graph_ref,
        Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>:$results);

    let assemblyFormat = [{
        $graph_ref `(` $operands `)` attr-dict
        `:` functional-type($operands, results)
    }];

    code extraClassDeclaration = [{
    /// Gets the Graph that this embedding references.
    Graph getGraph();

    //===------------------------------------------------------------------===//
    // CallOpInterface
    //===------------------------------------------------------------------===//

    CallInterfaceCallable getCallableForCallee() { return getGraphRefAttr(); }
    operand_range getArgOperands() { return getOperands(); }

    //===------------------------------------------------------------------===//
    // Graph
    //===------------------------------------------------------------------===//

    Block* getGraphContent()
    {
        if (auto graph = getGraph()) return graph.getGraphContent();
        return nullptr;
    }
    }];
}

//===----------------------------------------------------------------------===//
// SubgraphOp
//===----------------------------------------------------------------------===//

def DLNN_SubgraphOp : DLNN_Op<"subgraph", [
            HasParent<"NetworkOp, GraphOp">,
            HasOnlyGraphRegion,
            RegionKindInterface,
            SingleBlockImplicitTerminator<"OutputOp">,
            IsolatedFromAbove,
            DLNN_EnclaveOp,
            DLNN_Node,
            DLNN_Graph,
            OpAsmOpInterface,
            RecursiveSideEffects]> {
    let summary = "Separates a subgraph inside a graph";
    let description = [{
        The `dlnn.subgraph` operation declares its body to be an isolated sub-
        graph, separated from the surrounding graph.

        This is equivalent to exporting that subgraph to an outlined
        `dlnn.graph` and then embedding it using `dlnn.embed`, but happens in-
        place. However, this allows code motion between the parent and anonymous
        subgraphs.

        Example:
        ```mlir
        dlnn.graph @graph1(%arg0: !fm<3xf16[108,108]>) -> !fm<128xf16> {
            %result = dlnn.subgraph (%cap0 = %arg0 : !fm<3xf16[108,108]>) {
                // Implementation...
                dlnn.output %rel0 : !fm<128xf16>
            } -> !fm<128xf16>
            dlnn.output %result : !fm<128xf16>
        }
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$captures);
    let results = (outs Variadic<AnyType>:$results);
    let regions = (region SizedRegion<1>:$content);

    let hasCustomAssemblyFormat = 1;

    code extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // OpAsmOpInterface
    //===------------------------------------------------------------------===//

    static StringRef getDefaultDialect() { return "dlnn"; }
    }];
}

//===----------------------------------------------------------------------===//
// NodeOp
//===----------------------------------------------------------------------===//

def DLNN_NodeOp : DLNN_Op<"node", [
            HasParent<"NetworkOp, GraphOp, SubgraphOp">,
            SingleBlockImplicitTerminator<"OutputOp">,
            IsolatedFromAbove,
            DLNN_EnclaveOp,
            DLNN_Node,
            RecursiveSideEffects]> {
    let summary = "Represents an opaque node inside a graph";
    let description = [{
        The `dlnn.node` op defines an opaque terminal / node inside a graph.

        This operation is similar to `dlnn.subgraph`, except that it is not a
        traversible graph itself.

        Example:
        ```mlir
        dlnn.graph @graph1(%arg0: !fm<3xf16[108,108]>) -> !fm<128xf16> {
            %result = dlnn.node (%cap0 = %arg0 : !fm<3xf16[108,108]>) {
                // Implementation...
                dlnn.output %rel0 : !fm<128xf16>
            } -> !fm<128xf16>
            dlnn.output %result : !fm<128xf16>
        }
        ```
    }];

    let arguments = (ins Variadic<AnyType>);
    let results = (outs Variadic<AnyType>);
    let regions = (region SizedRegion<1>:$content);

    let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

def DLNN_OutputOp : DLNN_Op<"output", [
            HasParent<"NetworkOp, GraphOp, SubgraphOp">,
            NoSideEffect,
            DLNN_Node,
            Terminator,
            ReturnLike]> {
    let summary = "Defines the output value of a subgraph or node";
    let description = [{
        The `dlnn.output` operation serves as the terminator for DLNN operations
        that declare a region that produces result values.

        Example:
        ```mlir
        dlnn.graph @my_graph(%arg0: !fm<3xf16[108,108]>) -> !fm<128xf16> {
            // Implementation...
            dlnn.output %result : !fm<128xf16>
        }
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$operands);

    let assemblyFormat = [{ attr-dict ($operands^ `:` type($operands))?}];
}

//===----------------------------------------------------------------------===//
// ToTensorOp
//===----------------------------------------------------------------------===//

def DLNN_ToTensorOp : DLNN_Op<"to_tensor", [
            NoSideEffect,
            InferTypeOpInterface]> {
    let summary = "Converts an organized value into a ranked tensor";
    let description = [{
        The `dlnn.to_tensor` operation is a cast that converts an organized
        value with strong dimension semantics into a ranked tensor.

        This is the opposite of `dlnn.from_tensor`.

        This cast is a no-op, but forces the organized value to be materialized
        here for the chosen organization, adding a constraint to layout
        selection.

        Example:
        ```
        %tensor = dlnn.to_tensor %fm : !fm<3xf32[32,64], (C,W,H)[N]->(N,C,H,W)>
        // %tensor has type tensor<?x3x64x32>
        ```
    }];

    let arguments = (ins DLNN_OrganizedType:$organized);
    let results = (outs AnyRankedTensor:$tensor);

    let assemblyFormat = [{ attr-dict $organized `:` type($organized) }];

    // NOTE: We don't need a verifier since that task is performed by the
    //       InferTypeOpInterface verifier already!

    let hasFolder = 1;

    code extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // InferTypeOpInterface
    //===------------------------------------------------------------------===//

    static LogicalResult inferReturnTypes(
        MLIRContext*,
        Optional<Location>,
        ValueRange operands,
        DictionaryAttr,
        RegionRange,
        SmallVectorImpl<Type> &inferredReturnTypes)
    {
        if (operands.size() != 1) return failure();
        auto organized = operands[0].getType().dyn_cast<OrganizedType>();
        if (!organized) return failure();

        inferredReturnTypes.clear();
        inferredReturnTypes.push_back(organized.getTensorType());
        return success();
    }

    static bool isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs)
    {
        // NOTE: We do this to suppress a C++20 warning.
        return lhs.front() == rhs.front();
    }
    }];
}

//===----------------------------------------------------------------------===//
// FromTensorOp
//===----------------------------------------------------------------------===//

def DLNN_FromTensorOp : DLNN_Op<"from_tensor", [NoSideEffect]> {
    let summary = "Converts a ranked tensor into an organized value";
    let description = [{
        The `dlnn.from_tensor` operation is a cast that converts a ranked tensor
        into an organized value with strong dimension semantics by attaching an
        organization map.

        This is the opposite of `dlnn.to_tensor`.

        This cast is a no-op, but forces the organized value to be compatible
        with the chosen organization, adding a constraint to layout selection.

        Example:
        ```mlir
        // %tensor has type tensor<?x3x64x32>
        %fm = dlnn.from_tensor %tensor : !fm<3xf32[32,64], (C,W,H)[N]->(N,C,H,W)>
        ```
    }];

    let arguments = (ins AnyRankedTensor:$tensor);
    let results = (outs DLNN_OrganizedType:$organized);

    let hasCustomAssemblyFormat = 1;

    let hasVerifier = 1;
    let hasFolder = 1;
}

#endif
