// RUN: dlnn-opt %s --convert-dlnn-to-func | FileCheck %s

// CHECK: module {
module {
    // CHECK-LABEL: func.func @my_network(
    // CHECK-SAME: %[[arg0:.+]]: f64)
    // CHECK-NEXT: %[[call:.+]] = call @my_network_my_graph()
    // CHECK-NEXT: %[[square:.+]] = arith.mulf %[[call]], %[[call]]
    dlnn.network @my_network(%arg0: f64) -> f64 {
        // CHECK-LABEL: func.func private @my_network_my_graph(
        // CHECK-NEXT: %[[cst0:.+]] = arith.constant 1.000000e+00 : f64
        // CHECK-NEXT: return %[[cst0]]
        dlnn.graph @my_graph() -> f64 {
            %0 = arith.constant 1.0 : f64
            output %0 : f64
        }
        %call = embed @my_graph() : () -> (f64)
        %square = arith.mulf %call, %call : f64
        output %square : f64
    }

    // CHECK-LABEL: func.func @my_user(
    // CHECK-SAME: %[[arg0:.+]]: f64)
    func.func @my_user(%arg0: f64) -> f64 {
        // CHECK-NEXT: %[[result:.+]] = call @my_network(%arg0)
        %0 = dlnn.eval @my_network(%arg0) : (f64) -> (f64)
        // CHECK-NEXT: return %[[result]]
        return %0 : f64
    }
}
