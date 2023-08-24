// RUN: dlnn-opt %s --dlnn-flatten | FileCheck %s

// CHECK-LABEL: @my_network(
dlnn.network @my_network(%arg0: f64) -> f64 {
    // CHECK: %[[square:.+]] = arith.mulf
    %square = arith.mulf %arg0, %arg0 : f64
    // CHECK: %[[sum:.+]] = arith.addf %[[square]], %[[square]]
    %sum = subgraph (%c0 = %square : f64) {
        %sum = arith.addf %c0, %c0 : f64
        output %sum : f64
    } -> f64
    // CHECK: output %[[sum]]
    output %sum : f64
}
