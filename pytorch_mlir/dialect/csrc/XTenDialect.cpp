// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "XTenDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "XTenOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "XTenPasses.h"

using namespace mlir;
using namespace xilinx::xten;

void XTenDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "XTenOps.cpp.inc"
      >();
}

namespace xilinx {
namespace xten {

void registerXTenPasses() {
// #define GEN_PASS_REGISTRATION
  registerXTenToAffinePass();
  registerXTenDataflowPass();
  registerXTenNamePass();
}

}
}