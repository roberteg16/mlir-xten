//===- XTenMinimizeLiveTensors.h -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef XTenMinimizeLiveTensors_PASS_H
#define XTenMinimizeLiveTensors_PASS_H

#include <memory>
#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace xten {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createXTenMinimizeLiveTensorsPass();

} // namespace xten
} // namespace xilinx

#endif // XTenMinimizeLiveTensors_PASS_H