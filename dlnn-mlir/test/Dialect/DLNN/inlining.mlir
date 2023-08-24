// RUN: dlnn-opt %s -split-input-file --inline --dlnn-simplify | FileCheck %s

// CHECK-LABEL: @net0(
dlnn.network @net0() -> f64 {
    // CHECK-NOT: @g0(
    graph @g0() -> f64 {
        %0 = arith.constant 1.0 : f64
        output %0 : f64
    }
    // CHECK-NOT: @g1(
    graph @g1() -> f64 {
        %0 = arith.constant 2.0 : f64
        output %0 : f64
    }

    %0 = embed @g0() : () -> (f64)
    %1 = embed @g1() : () -> (f64)
    // CHECK: %[[result:.+]] = arith.constant 3.000000e+00 : f64
    %result = arith.addf %0, %1 : f64
    // CHECK: output %[[result]]
    output %result : f64
}

// -----

module {
    dlnn.network @net1() -> f64 {
        %result = arith.constant 3.0 : f64
        output %result : f64
    }

    // CHECK-LABEL: @fn(
    func.func @fn() -> f64 {
        // CHECK: %[[result:.+]] = dlnn.eval @net1()
        %result = dlnn.eval @net1() : () -> (f64)
        // CHECK: return %[[result]] : f64
        return %result : f64
    }
}
