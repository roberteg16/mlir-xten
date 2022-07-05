
// RUN: aten-opt %s -aten-visual-graph='operators-supported-path=%S/../../../lib/Transform/operators_supported.json' | diff %s.ref.json -

module attributes {torch.debug_module_name = "HelloWorld"} {
  func @forward(%arg0: tensor<1x16x64x64xf32>) -> tensor<1x16x64x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<16x16x3x3xf32>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<16xf32>
    %0 = linalg.init_tensor [1, 16, 64, 64] : tensor<1x16x64x64xf32>
    %1 = linalg.fused(%arg1 = %cst_1 : tensor<16x16x3x3xf32>, %arg2 = %0 : tensor<1x16x64x64xf32>, %arg3 = %cst_2 : tensor<16xf32>, %arg4 = %cst_0 : f32, %arg5 = %cst : f32, %arg6 = %arg0 : tensor<1x16x64x64xf32>) {
      %2 = tensor.pad %arg6 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg7: index, %arg8: index, %arg9: index, %arg10: index):
        tensor.yield %arg5 : f32
      } : tensor<1x16x64x64xf32> to tensor<1x16x66x66xf32>
      %3 = linalg.apply_bias_2d_fchw ins(%arg2, %arg3 : tensor<1x16x64x64xf32>, tensor<16xf32>) outs(%arg2 : tensor<1x16x64x64xf32>) -> tensor<1x16x64x64xf32>
      %4 = linalg.conv_2d_nchw_fchw {dilation = dense<1> : tensor<2xi64>, stride = dense<1> : tensor<2xi64>} ins(%2, %arg1 : tensor<1x16x66x66xf32>, tensor<16x16x3x3xf32>) outs(%3 : tensor<1x16x64x64xf32>) -> tensor<1x16x64x64xf32>
      %5 = linalg.lrelu_2d_nchw ins(%4, %arg4 : tensor<1x16x64x64xf32>, f32) outs(%4 : tensor<1x16x64x64xf32>) -> tensor<1x16x64x64xf32>
      linalg.yield %5 : tensor<1x16x64x64xf32>
    } -> tensor<1x16x64x64xf32>
    return %1 : tensor<1x16x64x64xf32>
  }
}
