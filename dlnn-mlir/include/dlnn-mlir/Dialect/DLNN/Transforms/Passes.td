//===- Passess.td - DLNN dialect passes --------------------*- tablegen -*-===//
//
// This is the definitions file for the DLNN dialect passes.
//
//===----------------------------------------------------------------------===//

#ifndef DLNN_PASSES
#define DLNN_PASSES

include "mlir/Pass/PassBase.td"

def DLNNFlatten : Pass<"dlnn-flatten", "NetworkOp"> {
    let summary = "Flatten DLNN graphs";
    let description = [{
        Flattens `dlnn.subgraph` hierarchies into their surrounding graph by
        inlining their contents. This never causes duplication of their child
        ops and is therefore applicable even in the presence of side-effects.

        Example:
        ```mlir
        dlnn.network @my_network(%arg0: f64) -> f64 {
            %square = arith.mulf %arg0, %arg0 : f64
            %sum = subgraph (%c0 = %square : f64) {
                %sum = arith.addf %c0, %c0 : f64
                output %sum : f64
            } -> f64
            output %sum : f64
        }
        ```
        is flattened to:
        ```mlir
        dlnn.network @my_network(%arg0: f64) -> f64 {
            %square = arith.mulf %arg0, %arg0 : f64
            %sum = arith.addf %square, %square : f64
            output %sum : f64
        }
        ```
    }];

    let constructor = "mlir::dlnn::createFlattenPass()";
}

def DLNNSimplify : Pass<"dlnn-simplify", "NetworkOp"> {
    let summary = "Simplify DLNN networks";
    let description = [{
        Simplifies `dlnn.network` contents using greedy rewrite patterns, which
        are semantically preserving but metadata destroying.

        Patterns:
            - RemoveUnusedCaptures

              Removes capture arguments from EnclaveOps that are unused.

              Example:
              ```mlir
              %2 = dlnn.subgraph (%arg0 = %0 : f64, %arg1 = %1 : f64) {
                  output %arg1 : f64
              } -> f64
              ```
              is simplified to:
              ```mlir
              %2 = dlnn.subgraph (%arg0 = %1 : f64) {
                  output %arg0 : f64
              } -> f64
              ```

            - RemoveUnusedReturns

              Removes EnclaveOp returns that are unused.

              Example:
              ```mlir
              dlnn.graph @my_graph(%0: f64) -> f64 {
                  %1, %2 = subgraph (%arg0 = %0 : f64) {
                      %0 = arith.addf %arg0, %arg0 : f64
                      %1 = arith.mulf %arg0, %arg0 : f64
                      output %0, %1 : f64, f64
                  } -> f64, f64
                  output %1 : f64
              }
              ```
              is simplified to:
              ```mlir
              dlnn.graph @my_graph(%0: f64) -> f64 {
                  %1 = subgraph (%arg0 = %0 : f64) {
                      %0 = arith.addf %arg0, %arg0 : f64
                      output %0 : f64
                  } -> f64
                  output %1 : f64
              }
              ```

            - RemoveDeadEmbeddings

              Removes `dlnn.embed` ops that are trivially dead.

              ```mlir
              dlnn.graph @my_graph(%0: f64) -> f64 {
                  %1 = arith.constant 1.0 : f64
                  %2 = dlnn.embed @other_graph(%0) : (f64) -> (f64)
                  output %1 : f64
              }
              ```
              assuming that @other_graph is side-effect free, is simplified to:
              ```mlir
              dlnn.graph @my_graph(%0: f64) -> f64 {
                  %1 = arith.constant 1.0 : f64
                  output %1 : f64
              }
              ```

            - RemoveDeadGraphs

              Removes `dlnn.graph` ops that have no uses.
    }];

    let constructor = "mlir::dlnn::createSimplifyPass()";
}

def ApplyTilingPass : Pass<"dlnn-apply-tiling", "func::FuncOp"> {
    let constructor = "mlir::dlnn::createApplyTilingPass()";

    let dependentDialects = [
        "scf::SCFDialect"
    ];
}

#endif
