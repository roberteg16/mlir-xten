// RUN: dlnn-opt %s -split-input-file --dlnn-apply-tiling --canonicalize | FileCheck %s

// CHECK-LABEL: @fuse_matmul_chain(
// CHECK-SAME: %[[arg0:.+]]: tensor<
func.func @fuse_matmul_chain(%arg0: tensor<8x?xf32>) -> tensor<8x?xf32> {
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c25 = arith.constant 25 : index
    %c24 = arith.constant 24 : index
    %c4 = arith.constant 4 : index
    // CHECK-DAG: %[[zero:.+]] = arith.constant 0.000000e+00 : f32
    %cst = arith.constant 0.000000e+00 : f32

    // CHECK: %[[fill:.+]] = linalg.fill ins(%[[zero]]
    %fill = linalg.fill
        ins(%cst : f32)
        outs(%arg0 : tensor<8x?xf32>) -> tensor<8x?xf32>

    // CHECK: %[[step1:.+]] = linalg.matmul
    // CHECK-SAME: ins(%[[arg0]], %[[arg0]] :
    // CHECK-SAME: outs(%[[fill]] :
    %step1 = linalg.matmul
        ins(%arg0, %arg0 : tensor<8x?xf32>, tensor<8x?xf32>)
        outs(%fill : tensor<8x?xf32>) -> tensor<8x?xf32>

    // CHECK: %[[step2]] = scf.for %[[iv]]
    %step2 = linalg.matmul {dlnn.tile_sizes=[2,2]}
        ins(%step1, %arg0 : tensor<8x?xf32>, tensor<8x?xf32>)
        outs(%fill : tensor<8x?xf32>) -> tensor<8x?xf32>

    return %step2 : tensor<8x?xf32>
}
