//===- XTenMinimizeLiveTensors.cpp -----------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This pass reorders operations to minimize the total size of live feature map
// tensors.

#include "PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "xten/Dialect/XTen/XTenDialect.h"
#include "xten/Dialect/XTen/XTenOps.h"
#include "xten/Util/Util.h"
#include "llvm/Support/Debug.h"
#include <memory>
#include <set>
#include <xten/Dialect/XTenNN/IR/XTenNNOps.h>

#define DEBUG_TYPE "xten-minimize-live"

using namespace mlir;
using namespace xilinx::xten;

namespace {

/// FM tensor sizes for a single operation.
struct OpSizes {
  /// Total size of the FM operands.
  size_t operands;
  /// Total size of the FM results.
  size_t results;
  /// Total size needed when executing, allowing for memory sharing.
  size_t running;
};

/// Information about memory requirements if the ops on a dependence
/// branch are executed.
/// Note that ops with multiple predecessors decide the order
/// of predecessor execution before they are added to the running
/// information, so once available it should be correct.
struct BranchRunning {
  /// The max size needed to run any operation on this branch.
  size_t maxRunning = 0;
  /// The size of the results of the last op in this branch.
  size_t lastResults = 0;
  /// The last op in this branch.
  Operation *lastOp;
};

/// Various analysis results about an operation.
struct OpInfo {
  /// The operand this node represents.
  Operation *const op;
  /// The feature map operand values of this node.
  SmallVector<Value> const operands;
  /// The feature map results this node.
  SmallVector<Value> const results;
  /// The value that will share memory with the result during execution, if any.
  Optional<Value> const sharesResultMemory = {};
  /// The consumers of any results. Note: this is filled progressively while
  /// collecting the operations.
  SmallVector<Operation *> consumers;
  /// Cumulative sizes of the FM tensors.
  OpSizes sizes = {};
  /// The preferred order of producers of the operands. Note: this is filled
  /// by post-analysis of the branching information.
  SmallVector<Operation *> orderedProducers;
  /// True when the operation has been moved in the IR.
  bool ordered = false;
};

/// HARDCODED returns the operand that will share memory with the result.

bool isXtenConvAddChained(Operation *op) {
  return isa<xilinx::xten::Conv2dTensorAddOp,
             xilinx::xten::Conv2dTensorAddReLUOp,
             xilinx::xten::Conv2dTensorAddLReLUOp,
             xilinx::xten::Conv2dTensorAddGlobalAveragePoolOp,
             xilinx::xten::Conv2dTensorAddReLUGlobalAveragePoolOp,
             xilinx::xten::Conv2dTensorAddLReLUGlobalAveragePoolOp>(op);
}

bool isInCoreChainSubgraph(Operation *op) {
  return op->hasAttr("Reason") &&
         op->getAttrOfType<StringAttr>("Reason") == "InCoreChain";
}

bool isConcatSubgraph(Operation *op) {
  return op->hasAttr("Op") && 
         op->getAttrOfType<StringAttr>("Op") == "Concat";
}

SmallVector<Value> getSubgraphIFMs(Operation *op) {

  // Handle IfmOperands attribute
  if (auto ifmIndices = op->getAttrOfType<ArrayAttr>("IfmOperands")) {

    // Get the operands from the values stored in IfmOperands Attr
    SmallVector<Value> ifmOperands;
    llvm::transform(ifmIndices.getAsValueRange<mlir::IntegerAttr>(),
                    std::back_inserter(ifmOperands), [&op](const APInt &idx) {
                      return op->getOperand(idx.getSExtValue());
                    });
    return ifmOperands;
  }
  return {};
}

Optional<Value> getSubgraphOFM(Operation *op) {

  // Handle OfmShare attribute
  if (auto ofmShare = op->getAttrOfType<mlir::IntegerAttr>("OfmShare")) {
    return {op->getOperand(ofmShare.getInt())};
  }
  return {};
}

/// HARDCODED returns the operand that will share memory with the result.
Optional<Value> sharesMemoryWithResult(Operation *op) {

  if (isInCoreChainSubgraph(op))
    return getSubgraphOFM(op);

  if (isXtenConvAddChained(op))
    return {op->getOperands().back()};
  return {};
}

/// HARDCODED returns all FM operands.
SmallVector<Value> getFmOperands(Operation *op) {
  // No input to the function.
  if (isa<func::FuncOp>(op))
    return {};

  // not sure of the syntax to unpack operand 0 - skip for now.
  assert(!isa<xilinx::xten::ConcatOp>(op));

  // Per operand defined IFMs.
  if (isa<xilinx::xten::AddOp, xilinx::xten::MulOp, xilinx::xten::MMOp>(op))
    return op->getOperands();

  if (isXtenConvAddChained(op))
    return {op->getOperands().front(), op->getOperands().back()};

  if (isInCoreChainSubgraph(op))
    return getSubgraphIFMs(op);

  if (isConcatSubgraph(op))
    return op->getOperands();

  // torch.aten.cat requires supporting torch.prim.ListConstruct,
  // which constructs a tensor concatenating all of its operands.
  // The operation produces a torch.list<vtensor> of tensors and
  // is used as the only operand of a torch.aten.cat.
  // torch.aten.cat needs no additional treatment since only
  // the first operand (the torch.prim.ListConstruct) is used.
  if (isa<torch::Torch::PrimListConstructOp>(op)) {
    return op->getOperands();
  }

  // TODO: there is no guarantee that FM is only the 1st operand. It
  // would be better to check all ops, preferably via an interface.
  // Okay for prototype, knowing this may backfire in debug effort.
  return {op->getOperand(0)};
}

/// Return the size of the tensor type of \p val.
/// It is an error to call this with a non-tensor typed value.
size_t getSize(Value val) {
  auto type = val.getType();
  if (isa<torch::Torch::BaseTensorType>(type)) {
    return xilinx::xten::getTensorVolume(val.getType());
  }
  if (isa<ShapedType>(type)) {
    return cast<ShapedType>(type).getSizeInBits();
  }
  // Otherwise, this is a torch.list<vtensor> that was used
  // to construct a list of tensors to be used in a torch.aten.cat
  // operation.
  // It is safe to return 0 for it since the size will be attached
  // to the operands of torch.prim.ListConstruct and to the result
  // of the torch.aten.cat operation.
  assert(isa<torch::Torch::ListType>(type));
  return 0;
}

/// Debugging support - returns a simple name for an op.
StringRef getName(Operation *op) {
  if (op->hasAttr("layer_name"))
    return op->getAttrOfType<StringAttr>("layer_name").getValue();
  return op->getName().getStringRef();
}

/// Debugging support - returns a string with all the op names.
std::string toStr(SmallVector<Operation *> const &vec) {
  std::string str("(");
  for (auto *op : vec)
    str += std::string(::getName(op)) + " ";
  return str + ")";
}

/// Determine the in/out/running L2 memory needed per Fwd.
void setOpSizes(OpInfo &opInfo) {
  size_t outgoing = 0;
  for_each(opInfo.results,
           [&outgoing](Value val) { outgoing += getSize(val); });
  opInfo.sizes.results = outgoing;
  size_t incoming = 0;
  for_each(opInfo.operands,
           [&incoming](Value val) { incoming += getSize(val); });
  opInfo.sizes.operands = incoming;
  opInfo.sizes.running = incoming;
  if (!opInfo.sharesResultMemory)
    opInfo.sizes.running += outgoing;
}

class XTenMinimizeLiveTensorsPass
    : public XTenMinimizeLiveTensorsBase<XTenMinimizeLiveTensorsPass> {
public:
  XTenMinimizeLiveTensorsPass() = default;
  XTenMinimizeLiveTensorsPass(const XTenMinimizeLiveTensorsPass &pass) =
      default;

  // Recursively collect the OpInfo of all FM producers.
  void collectOperandInfo(OpInfo const &opInfo) { // NOLINT(misc-no-recursion)
    // Visit all FM operands to collect their OpInfo.
    for (auto operand : opInfo.operands) {
      Operation *defOp = operand.getDefiningOp();
      if (defOp == nullptr) {
        // Use currFn as stand-in for BlockArguments.
        assert(operand.isa<BlockArgument>());
        defOp = currFn;
      }

      // If OpInfo is already created, so we only need to note this consumer.
      auto prevInfoIt = opToInfo.find(defOp);
      if (prevInfoIt != opToInfo.end()) {
        prevInfoIt->second.consumers.push_back(opInfo.op);
        continue;
      }

      // Create the new OpInfo for the operand.
      SmallVector<Value> const fmOperands = getFmOperands(defOp);
      SmallVector<Value> fmResults;
      if (defOp != currFn) {
        fmResults = defOp->getResults();
      } else {
        fmResults = SmallVector<Value>(currFn.getBody().front().getArguments());
      }
      Optional<Value> const sharesResultMemory = sharesMemoryWithResult(defOp);
      OpInfo info = {.op = defOp,
                     .operands = fmOperands,
                     .results = fmResults,
                     .sharesResultMemory = sharesResultMemory,
                     .consumers = {opInfo.op}};
      auto [opFwdIt, succeeded] = opToInfo.emplace(defOp, std::move(info));
      setOpSizes(opFwdIt->second);
      // Recursively collect details of the operands of this operand.
      collectOperandInfo(opFwdIt->second);
    }
  }

  /// Checks for illegal dead code by traversing operations and
  /// verifying if they are dead and any of their operands match any OpInfo.
  /// Dead code can be legal if it is not connected to operation collection
  /// through 'OpInfo' data structure.
  bool hasIllegalDeadCode() {
    return llvm::any_of(currFn.getBody().getOps(), [&](Operation &op) {
      if (opToInfo.find(&op) != opToInfo.end())
        return false; // okay as we schedule this op

      return llvm::any_of(op.getOperands(), [&](Value value) {
        // okay so long as it doesn't use the output of a scheduled op
        auto *defOp = value.getDefiningOp();
        if (defOp != nullptr && opToInfo.find(defOp) != opToInfo.end()) {
          llvm::errs() << "Unknown operation: ";
          defOp->dump();
          return true;
        }
        return false;
      });
    });
  }

  /// Recursively determine branch running sizes.
  ///
  /// \p opInfo points to the info for the op being analyzed.
  /// \p brInfo incoming branch info, to be updated by this op.
  /// \p completed contains the BranchRunning info for any fully
  ///    analyzed operations.
  void determineBranchRunning(OpInfo *opInfo, // NOLINT(misc-no-recursion)
                              BranchRunning &brInfo,
                              std::map<Operation *, BranchRunning> &completed) {
    // Analyze simple fallthrough operations.
    while (opInfo->operands.size() < 2 && opInfo->consumers.size() < 2) {
      brInfo.maxRunning = std::max(brInfo.maxRunning, opInfo->sizes.results);
      brInfo.lastResults = opInfo->sizes.results;
      opInfo->orderedProducers = {brInfo.lastOp};
      brInfo.lastOp = opInfo->op;
      completed.insert({opInfo->op, brInfo});

      if (opInfo->consumers.empty())
        return; // Nothing more to compute.
      opInfo = &opToInfo.at(opInfo->consumers.front());
    }

    // At a joining point - collect this branch and proceed iff all incoming
    // branches have been collected.
    SmallVector<BranchRunning> branches;
    for (Value val : opInfo->operands) {
      Operation *op = val.getDefiningOp();
      if (op == nullptr)
        op = currFn; // BlockArgument stand-in.

      auto opIt = completed.find(op);
      if (opIt == completed.end())
        return; // Another producer needs to complete first.
      branches.push_back(opIt->second);
    }

    std::sort(branches.begin(), branches.end(),
              [](BranchRunning &aBranch, BranchRunning &bBranch) -> bool {
                return (aBranch.maxRunning - aBranch.lastResults) >
                       (bBranch.maxRunning - bBranch.lastResults);
              });
    opInfo->orderedProducers.clear();
    for (BranchRunning const &branch : branches)
      opInfo->orderedProducers.push_back(branch.lastOp);

    // Complete the brInfo for this operation.
    size_t maxRunning = opInfo->sizes.running;
    for (BranchRunning const &branch : branches)
      maxRunning = std::max(maxRunning, branch.maxRunning);
    brInfo.maxRunning = maxRunning;
    brInfo.lastResults = opInfo->sizes.results;
    brInfo.lastOp = opInfo->op;
    completed.insert({opInfo->op, brInfo});

    // Continue to all consumers.
    for (Operation *consumer : opInfo->consumers) {
      BranchRunning nextRunning{.maxRunning = maxRunning,
                                .lastResults = opInfo->sizes.results,
                                .lastOp = opInfo->op};
      determineBranchRunning(&opToInfo.at(consumer), nextRunning, completed);
    }
    assert(opInfo->orderedProducers.size() == opInfo->operands.size());
  }

  /// Move the operators to the desired lexical order.
  void moveToOrder(OpInfo const &fwd) { // NOLINT(misc-no-recursion)
    for (Operation *op : fwd.orderedProducers) {
      if (op == currFn)
        continue; // BlockArguments cannot be moved.
      OpInfo &visitFwd = opToInfo.at(op);
      if (visitFwd.ordered)
        continue;

      op->moveBefore(fwd.op);
      visitFwd.ordered = true;
      moveToOrder(visitFwd);
    }
  }

  void runOnOperation() override {
    auto fwdFn = getOperation();
    mlir::Region &body = fwdFn.getBody();
    if (!body.hasOneBlock()) {
      fwdFn.emitError("function has complex control flow, aborting");
      signalPassFailure();
      return;
    }
    currFn = fwdFn;

    // A single block is expected. Building the initial graph starts from the
    // return and is successful when all FMs are ultimately produced from the
    // function arguments.
    Operation *returnStmt = body.begin()->getTerminator();
    assert(isa<func::ReturnOp>(returnStmt) &&
           "A function must terminate with a return stmt");
    SmallVector<Value> const retVal = returnStmt->getOperands();
    OpInfo fwdInfo = {.op = returnStmt, .operands = retVal, .results = {}};
    auto [opFwdIt, succeeded] =
        opToInfo.emplace(returnStmt, std::move(fwdInfo));
    OpInfo const &retFwd = opFwdIt->second;

    collectOperandInfo(retFwd);

    auto prevFwdIt = opToInfo.find(currFn);
    if (prevFwdIt == opToInfo.end()) {
      returnStmt->emitError(
          "function entry is not reached from the return stmt");
      signalPassFailure();
      return;
    }
    OpInfo &fnFwd = prevFwdIt->second;

    // Checks for any illegal dead code in currFn
    if (hasIllegalDeadCode()) {
      fwdFn->emitError("function cannot be rescheduled due to illegal dead "
                       "code, aborting.\n");
      signalPassFailure();
      return;
    }

    BranchRunning nextRunning{.maxRunning = fnFwd.sizes.running,
                              .lastResults = fnFwd.sizes.results,
                              .lastOp = fnFwd.op};
    std::map<Operation *, BranchRunning> completed;
    determineBranchRunning(&fnFwd, nextRunning, completed);

    LLVM_DEBUG(print());

    moveToOrder(retFwd);
  }

private:
  /// The analysis results for each operation.
  std::map<Operation *, OpInfo> opToInfo;
  /// The function being analyzed - needed in places to represent
  /// BlockArguments.
  mlir::func::FuncOp currFn;

  /// Debugging support - print some details of all .
  void print() {
    for (auto &[op, info] : opToInfo) {
      llvm::errs() << ::getName(op)
                   << " producers: " << toStr(info.orderedProducers)
                   << " consumers: " << toStr(info.consumers) << "\n";
    }
    llvm::errs() << "----\n";
  }
};

} // namespace

namespace xilinx::xten {

std::unique_ptr<OperationPass<func::FuncOp>>
createXTenMinimizeLiveTensorsPass() {
  return std::make_unique<XTenMinimizeLiveTensorsPass>();
}

} // namespace xilinx::xten
