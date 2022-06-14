//===- aten_resnet_block.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aten-opt %s -aten-to-xten -split-input-file | FileCheck %s

// --- conv2d_tensoradd_branch_relu

// CHECK-LABEL:  func @forward_conv2d_tensoradd_branch_relu

// CHECK: %[[CONV1:.*]] = "xten.conv2d_relu"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONVSKIP:.*]] = "xten.conv2d"(%[[CONV1]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONV2:.*]] = "xten.conv2d_relu"(%[[CONV1]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONV3:.*]] = "xten.conv2d_relu"(%[[CONV2]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,64,64],f32>
// CHECK: %[[CONVADD:.*]] = "xten.conv2d_tensoradd_relu"(%[[CONV3]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[CONVSKIP]]) : (!torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.vtensor<[1,256,64,64],f32>) -> !torch.vtensor<[1,256,64,64],f32>

func @forward_conv2d_tensoradd_branch_relu(%arg0: !torch.vtensor<[1,256,64,64],f32>) -> !torch.vtensor<[1,256,64,64],f32> {
  %0 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256xf32>) : !torch.vtensor<[256],f32>
  %1 = torch.vtensor.literal(dense<1.000000e+00> : tensor<256x256x1x1xf32>) : !torch.vtensor<[256,256,1,1],f32>
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.conv2d %arg0, %1, %0, %2, %3, %2, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %5 = torch.aten.relu %4 : !torch.vtensor<[1,256,64,64],f32> -> !torch.vtensor<[1,256,64,64],f32>
  %6 = torch.aten.conv2d %5, %1, %0, %2, %3, %2, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %7 = torch.aten.conv2d %5, %1, %0, %2, %3, %2, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %8 = torch.aten.relu %7 : !torch.vtensor<[1,256,64,64],f32> -> !torch.vtensor<[1,256,64,64],f32>
  %9 = torch.aten.conv2d %8, %1, %0, %2, %3, %2, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %10 = torch.aten.relu %9 : !torch.vtensor<[1,256,64,64],f32> -> !torch.vtensor<[1,256,64,64],f32>
  %11 = torch.aten.conv2d %10, %1, %0, %2, %3, %2, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[256,256,1,1],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %12 = torch.aten.add.Tensor %6, %11, %int1 : !torch.vtensor<[1,256,64,64],f32>, !torch.vtensor<[1,256,64,64],f32>, !torch.int -> !torch.vtensor<[1,256,64,64],f32>
  %13 = torch.aten.relu %12 : !torch.vtensor<[1,256,64,64],f32> -> !torch.vtensor<[1,256,64,64],f32>
  return %13 : !torch.vtensor<[1,256,64,64],f32>
}
