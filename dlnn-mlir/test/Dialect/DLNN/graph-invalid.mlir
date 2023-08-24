// RUN: dlnn-opt %s -split-input-file -verify-diagnostics

dlnn.network @net0() {
    // expected-error@+1 {{function signature}}
    "dlnn.graph"() ({
        ^bb0(%arg0: f32):
            %0 = arith.constant 0.0 : f64
            output %0 : f64
    }) {function_type = (f64) -> f64, sym_name = "wrong_block_args"} : () -> ()
    output
}

// -----

dlnn.network @net1() {
    graph @wrong_terminator() -> f64 {
        %0 = arith.constant 0.0 : f64
        // expected-error@+1 {{must be a Node}}
        func.return %0 : f64
    }
    output
}

// -----

dlnn.network @net2() {
    // expected-error@+1 {{does not reference}}
    embed @non_existant() : () -> ()
    output
}

// -----

dlnn.network @net3() {
    graph @recursion() -> f64 {
        // expected-error@+1 {{recursive embedding}}
        %0 = embed @recursion() : () -> (f64)
        output %0 : f64
    }
    output
}

// -----

dlnn.network @net4() {
    graph @wrong_input(%arg0: f32) -> f64 {
        %0 = arith.constant 0.0 : f64
        output %0 : f64
    }
    %0 = arith.constant 0.0 : f64
    // expected-error@+1 {{does not match graph input type}}
    %1 = embed @wrong_input(%0) : (f64) -> (f64)
    output
}

// -----

dlnn.network @net5() {
    graph @wrong_output() -> f32 {
        %0 = arith.constant 0.0 : f32
        output %0 : f32
    }
    // expected-error@+1 {{does not match graph output type}}
    %1 = embed @wrong_output() : () -> (f64)
    output
}
