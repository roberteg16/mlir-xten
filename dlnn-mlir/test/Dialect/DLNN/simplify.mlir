// RUN: dlnn-opt %s -split-input-file --dlnn-simplify | FileCheck %s

// CHECK-LABEL: @net0(
dlnn.network @net0() ->f64 {
    // CHECK-DAG: @unused_capture(
    graph @unused_capture() -> f64 {
        %unused = arith.constant 0.0 : f64
        // CHECK: %[[result:.+]] = subgraph () {
        %result = subgraph (%0 = %unused : f64) {
            %1 = arith.constant 1.0 : f64
            output %1 : f64
        // CHECK: } -> f64
        } -> f64
        // CHECK: output %[[result]] : f64
        output %result : f64
    }

    %0 = embed @unused_capture() : () -> (f64)
    output %0 : f64
}

// -----

// CHECK-LABEL: @net1(
dlnn.network @net1() {
    // CHECK-DAG: @unused_return(
    graph @unused_return() -> f64 {
        // CHECK: %[[r0:.+]] = subgraph () {
        %r0, %r1 = subgraph () {
            // CHECK: %[[cst0:.+]] = arith.constant 0.000000e+00 : f64
            %cst0 = arith.constant 0.0 : f64
            %cst1 = arith.constant 1.0 : f64
            // CHECK: output %[[cst0]] : f64
            output %cst0, %cst1 : f64, f64
        // CHECK: } -> f64
        } -> f64, f64
        // CHECK: output %[[r0]] : f64
        output %r0 : f64
    }

    %0 = embed @unused_return() : () -> (f64)
    output %0 : f64
}

// -----

// CHECK-LABEL: @net2(
dlnn.network @net2() {
    // CHECK-NOT: @dead(
    graph @dead() -> f64 {
        %cst0 = arith.constant 0.0 : f64
        output %cst0 : f64
    }

    // CHECK-DAG: @dead_embedding(
    graph @dead_embedding() -> f64 {
        // CHECK: %[[result:.+]] = arith.constant 0.000000e+00 : f64
        %result = arith.constant 0.0 : f64
        // CHECK-NOT: embed @dead
        %dead = embed @dead() : () -> (f64)
        // CHECK: output %[[result]] : f64
        output %result : f64
    }

    // CHECK-NOT: @dead(

    %0 = embed @dead_embedding() : () -> (f64)
    output %0 : f64
}
