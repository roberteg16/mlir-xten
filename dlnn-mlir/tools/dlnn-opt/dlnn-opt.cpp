/// Main entry point for the dlnn-mlir optimizer driver.
///
/// @file
/// @author      Karl F. A. Friebel (karl.friebel@amd.com)

#include "dlnn-mlir/Conversion/Passes.h"
#include "dlnn-mlir/Dialect/DLNN/IR/DLNN.h"
#include "dlnn-mlir/Dialect/DLNN/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

int main(int argc, char* argv[])
{
    registerAllPasses();

    DialectRegistry registry;
    registerAllDialects(registry);

    registry.insert<dlnn::DLNNDialect>();
    mlir::dlnn::registerDLNNPasses();
    mlir::dlnn::registerDLNNConversionPasses();

    return asMainReturnCode(
        MlirOptMain(argc, argv, "dlnn-mlir optimizer driver\n", registry));
}
