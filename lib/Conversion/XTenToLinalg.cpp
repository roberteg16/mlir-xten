//===- XTenToLinalg.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "xten/Conversion/XTenToLinalgPass.h"
#include "xten/Dialect/XTen/XTenDialect.h"
#include "xten/Dialect/XTen/XTenOps.h"
#include "xten/Util/Util.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "xten-to-linalg-pass"

using namespace mlir;
using namespace xilinx::xten;
using namespace mlir::torch;

static Value applyPad(Location loc, Value input, ArrayRef<int64_t> pad,
                      Attribute padAttr, OpBuilder &rewriter) {
  // Input should be padded if necessary.
  if (llvm::all_of(pad, [](int64_t p) { return p == 0; }))
    return input;

  ShapedType inputTy = input.getType().cast<ShapedType>();
  Type inputETy = inputTy.getElementType();
  auto inputShape = inputTy.getShape();

  assert(inputShape.size() == pad.size());

  SmallVector<int64_t, 4> paddedShape;
  SmallVector<OpFoldResult, 8> lowIndices;
  SmallVector<OpFoldResult, 8> highIndices;
  for (int i = 0, s = inputShape.size(); i < s; i++) {
    auto lowPad = pad[i];
    auto highPad = pad[i];
    paddedShape.push_back(inputShape[i] + highPad + lowPad);
    lowIndices.push_back(rewriter.getIndexAttr(lowPad));
    highIndices.push_back(rewriter.getIndexAttr(highPad));
  }

  Value padValue = rewriter.create<arith::ConstantOp>(loc, padAttr);

  return tensor::createPadScalarOp(
             RankedTensorType::get(paddedShape, inputETy), input, padValue,
             lowIndices, highIndices, /*nofold=*/false, loc, rewriter)
      .result();
}

/// Return a zero-initialized tensor of given size and dtype.
static Value zeroInit(ArrayRef<int64_t> sizes, mlir::Type elementType, Location loc,
                         ConversionPatternRewriter &rewriter) {
  Value initTensor = rewriter.create<linalg::InitTensorOp>(loc, sizes, elementType);
  Value c0float = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(elementType, 0.0));
  return rewriter.create<linalg::FillOp>(loc, c0float, initTensor)
      .getResult(0);
}

/// Return an aten bias (vtensor or none) converted to a standard bias tensor.
static Value convertBias(Operation *op, Value atenBias, Location loc,
                         ConversionPatternRewriter &rewriter) {
  if (atenBias.getType().isa<Torch::NoneType>()) {
    auto resultTy= op->getResult(0).getType().dyn_cast<torch::Torch::BaseTensorType>();
    return zeroInit(resultTy.getSizes()[1], resultTy.getDtype(), loc, rewriter);
  }
  return ToBuiltinTensorTypeCast(rewriter, atenBias);
}

/// Produces an output-dimensioned tensor, initialized with an aten bias (vtensor or none).
static Value getBiasedInit(Operation *op, Value atenBias, Location loc,
                         ConversionPatternRewriter &rewriter) {
  auto outputTy =
      op->getResult(0).getType().dyn_cast<torch::Torch::BaseTensorType>();
  assert(outputTy);
  auto elementType = outputTy.getDtype();
  Value initTensor = rewriter.create<linalg::InitTensorOp>(
      loc, outputTy.getSizes(), elementType);

  if (atenBias.getType().isa<Torch::NoneType>()) {
    Value c0float = rewriter.create<arith::ConstantOp>(
        loc, FloatAttr::get(elementType, 0.0));
    return rewriter.create<linalg::FillOp>(loc, c0float, initTensor)
        .getResult(0);
  }
  auto bias = ToBuiltinTensorTypeCast(rewriter, atenBias);
  auto biasType = bias.getType().cast<RankedTensorType>();
  assert (biasType.getRank() == 1);
  assert (elementType == biasType.getElementType());

  auto resultRank = initTensor.getType().cast<RankedTensorType>().getRank();
  SmallVector<AffineMap> indexingMaps = {
      // bias is used to initialize the channels - dimension 1 of output
      AffineMap::get(/*dimCount=*/resultRank, /*symbolCount=*/0,
                     rewriter.getAffineDimExpr(1), rewriter.getContext()),
      rewriter.getMultiDimIdentityMap(resultRank)};
  SmallVector<StringRef> iteratorTypes(resultRank, "parallel");
  return rewriter
             .create<linalg::GenericOp>(
                 loc, initTensor.getType(), bias, initTensor, indexingMaps,
                 iteratorTypes,
                 [](OpBuilder &b, Location loc, ValueRange args) {
                   b.create<linalg::YieldOp>(loc, args[0]);
                 })
             .getResult(0);
}

template<class T>
static LogicalResult processConv2d(T &conv2dOp, Location &loc, Value &input,
                                  Type &elementType, Operation *op, 
                                  ConversionPatternRewriter &rewriter) {
  if (!elementType.isa<mlir::FloatType>())
    return op->emitError("unimplemented: non-floating point type");

  SmallVector<int64_t> paddingInts;
  paddingInts.resize(2, 0);
  if (!matchPattern(conv2dOp.padding(),
                    Torch::m_TorchConstantIntList(paddingInts))) {
    return rewriter.notifyMatchFailure(
        op, "only support constant padding values");
  }

  /// paddedInput. input shape change based on padding
  Attribute zeroAttr = rewriter.getZeroAttr(elementType);
  input = applyPad(loc, input, paddingInts, zeroAttr, rewriter);

  int64_t groups;
  if (!matchPattern(conv2dOp.groups(), Torch::m_TorchConstantInt(&groups)))
    return rewriter.notifyMatchFailure(op, "only support constant int group");

  if (groups != 1)
    return op->emitError("Only support groups value '1'");

  return success();
}

namespace {

template <class T>
class XTenBinaryOpConversion : public ConversionPattern {
public:
  XTenBinaryOpConversion(StringRef rootName, PatternBenefit benefit,
                         MLIRContext *ctx)
      : ConversionPattern(rootName, benefit, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    Value A = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value B = ToBuiltinTensorTypeCast(rewriter, operands[1]);

    auto tensorTy = operands[0].getType().cast<Torch::BaseTensorType>();
    auto elementTy = tensorTy.getDtype();
    auto sizes = tensorTy.getSizes();
    auto rank = sizes.size();
    Value C = rewriter.create<linalg::InitTensorOp>(loc, sizes, elementTy);

    SmallVector<Value, 2> inputTensors{A, B};
    SmallVector<Value, 1> outputTensors{C};

    auto identMap = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap, 4> indexMap(3, identMap);

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, C.getType(), inputTensors, outputTensors, indexMap,
        SmallVector<StringRef>(rank, getParallelIteratorTypeName()), "",
        static_cast<const T *>(this)->getDefaultLibraryFunc(),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          auto result = static_cast<const T *>(this)->emitBinaryOp(
              op, elementTy, rewriter, blockArgs[0], blockArgs[1]);
          nestedBuilder.create<linalg::YieldOp>(loc, result);
        }).getResult(0);

    auto torchTensorCast =
        ToTorchTensorTypeCast(rewriter, linalgOp, op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenAddOpConversion : public XTenBinaryOpConversion<XTenAddOpConversion> {
public:
  explicit XTenAddOpConversion(MLIRContext *context)
      : XTenBinaryOpConversion(AddOp::getOperationName(), 1, context) {}

  StringRef getDefaultLibraryFunc() const {
    return "xten_add_op";
  }

  Value emitBinaryOp(Operation *op, Type elementTy,
                     ConversionPatternRewriter &rewriter, Value a,
                     Value b) const {
    if (elementTy.isa<FloatType>())
      return rewriter.create<mlir::arith::AddFOp>(op->getLoc(), a, b);
    else
      return rewriter.create<mlir::arith::AddIOp>(op->getLoc(), a, b);
  }
};

class XTenMulOpConversion : public XTenBinaryOpConversion<XTenMulOpConversion> {
public:
  explicit XTenMulOpConversion(MLIRContext *context)
      : XTenBinaryOpConversion(MulOp::getOperationName(), 1, context) {}

  StringRef getDefaultLibraryFunc() const {
    return "xten_mul_op";
  }

  Value emitBinaryOp(Operation *op, Type elementTy,
                     ConversionPatternRewriter &rewriter, Value a,
                     Value b) const {
    if (elementTy.isa<FloatType>())
      return rewriter.create<mlir::arith::MulFOp>(op->getLoc(), a, b);
    else
      return rewriter.create<mlir::arith::MulIOp>(op->getLoc(), a, b);
  }
};

class XTenMMOpConversion : public ConversionPattern {
public:
  explicit XTenMMOpConversion(MLIRContext *context)
      : ConversionPattern(MMOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto mmult = cast<MMOp>(op);
    auto loc = mmult.getLoc();

    auto resultTy = op->getResult(0).getType();
    auto tTy = resultTy.cast<Torch::BaseTensorType>();
    auto oper0Ty = operands[0].getType().cast<Torch::BaseTensorType>();
    auto oper1Ty = operands[1].getType().cast<Torch::BaseTensorType>();
    std::vector<int64_t> sizes{oper0Ty.getSizes()[0], oper1Ty.getSizes()[1]};

    Value A = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value B = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value C = zeroInit(tTy.getSizes(), tTy.getDtype(), loc, rewriter);

    auto mulOp = rewriter
        .create<linalg::MatmulOp>(loc, C.getType(), ValueRange{A, B},
                                  ValueRange{C})
        .getResult(0);

    auto tensor_cast =
        ToTorchTensorTypeCast(rewriter, mulOp, resultTy);
    rewriter.replaceOp(op, tensor_cast);

    return success();
  }
};

class XTenConv2dOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2d = cast<Conv2dOp>(op);
    auto loc = conv2d.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value biasInitTensor = getBiasedInit(op, conv2d.bias(), loc, rewriter);

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op->emitError("unimplemented: non-floating point type");

    SmallVector<int64_t> paddingInts;
    paddingInts.resize(2, 0);
    if (!matchPattern(conv2d.padding(),
                      Torch::m_TorchConstantIntList(paddingInts))) {
      return rewriter.notifyMatchFailure(
          op, "only support constant padding values");
    }

    // paddedInput. input shape change based on padding
    Attribute zeroAttr = rewriter.getZeroAttr(elementType);
    input = applyPad(loc, input, paddingInts, zeroAttr, rewriter);

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2d.stride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2d.dilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    int64_t groups;
    if (!matchPattern(conv2d.groups(), Torch::m_TorchConstantInt(&groups)))
      return rewriter.notifyMatchFailure(op, "only support constant int group");

    if (groups != 1)
      return op->emitError("Only support groups value '1'");

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);

    Value conv2dVal =
        rewriter
            .create<linalg::Conv2DNchwFchwOp>(
                loc, biasInitTensor.getType(), ValueRange{input, weight},
                biasInitTensor, stridesAttr, dilationAttr)
            .getResult(0);

    auto torchTensorCast =
        ToTorchTensorTypeCast(rewriter, conv2dVal, op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dReluOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dReluOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dReLUOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2dRelu = cast<Conv2dReLUOp>(op);
    auto loc = conv2dRelu.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    LogicalResult result = processConv2d(conv2dRelu, loc, input, elementType, 
        op, rewriter);
    if (result.failed())
      return result;

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2dRelu.stride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2dRelu.dilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultTensorType.getShape(), elementType);

    Value conv2dReluVal =
        rewriter
            .create<linalg::Conv2DReluOp>(loc, initTensor.getType(),
                                          ValueRange{input, weight, bias},
                                          initTensor, stridesAttr, dilationAttr)
            .getResult(0);

    if (op->hasAttr("layer_name")) {
      auto attrVal = op->getAttr("layer_name").cast<StringAttr>();
      Operation *conv2dReluValOps = conv2dReluVal.getDefiningOp();
      conv2dReluValOps->setAttr(llvm::StringRef("layer_name"), attrVal);
    }

    auto torchTensorCast = ToTorchTensorTypeCast(rewriter, conv2dReluVal,
                                                 op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dLeakyReluOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dLeakyReluOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dLReLUOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2dLRelu = cast<Conv2dLReLUOp>(op);
    auto loc = conv2dLRelu.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    if (!isa<Torch::ConstantFloatOp>(operands[7].getDefiningOp()))
      return op->emitError("Alpha, unimplemented: non-floating point type");

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    LogicalResult result = processConv2d(conv2dLRelu, loc, input, elementType, 
        op, rewriter);
    if (result.failed())
      return result;

    // Getting alpha value
    auto c = cast<Torch::ConstantFloatOp>(operands[7].getDefiningOp()).value();
    auto ty = rewriter.getF32Type();
    auto add_const = rewriter.getFloatAttr(ty, c.convertToDouble());
    Value alpha = rewriter.create<arith::ConstantOp>(loc, ty, add_const);

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2dLRelu.stride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2dLRelu.dilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultTensorType.getShape(), elementType);

    Value conv2dLReluVal = rewriter
                               .create<linalg::Conv2DLreluOp>(
                                   loc, initTensor.getType(),
                                   ValueRange{input, weight, bias, alpha},
                                   initTensor, stridesAttr, dilationAttr)
                               .getResult(0);

    if (op->hasAttr("layer_name")) {
      auto attrVal = op->getAttr("layer_name").cast<StringAttr>();
      Operation *conv2dLReluValOps = conv2dLReluVal.getDefiningOp();
      conv2dLReluValOps->setAttr(llvm::StringRef("layer_name"), attrVal);
    }

    auto torchTensorCast = ToTorchTensorTypeCast(rewriter, conv2dLReluVal,
                                                 op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dTensorAddOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dTensorAddOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dTensorAddOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2d = cast<Conv2dTensorAddOp>(op);
    auto loc = conv2d.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    LogicalResult result = processConv2d(conv2d, loc, input, elementType, 
        op, rewriter);
    if (result.failed())
      return result;

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2d.stride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2d.dilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultTensorType.getShape(), elementType);

    // Get add input feature map
    Value add_ifm = ToBuiltinTensorTypeCast(rewriter, operands[7]);

    // Change appropriate operation over here
    Value conv2dVal = rewriter
                               .create<linalg::Conv2DTensorAddOp>(
                                   loc, initTensor.getType(),
                                   ValueRange{input, add_ifm, weight, bias}, // add_ifm should be in ValueRange
                                   initTensor, stridesAttr, dilationAttr)
                               .getResult(0);

    if (op->hasAttr("layer_name")) {
      auto attrVal = op->getAttr("layer_name").cast<StringAttr>();
      Operation *conv2dValOps = conv2dVal.getDefiningOp();
      conv2dValOps->setAttr(llvm::StringRef("layer_name"), attrVal);
    }

    auto torchTensorCast = ToTorchTensorTypeCast(rewriter, conv2dVal,
                                                 op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dTensorAddReLUOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dTensorAddReLUOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dTensorAddReLUOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2dRelu = cast<Conv2dTensorAddReLUOp>(op);
    auto loc = conv2dRelu.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    LogicalResult result = processConv2d(conv2dRelu, loc, input, elementType, 
        op, rewriter);
    if (result.failed())
      return result;

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2dRelu.stride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2dRelu.dilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultTensorType.getShape(), elementType);

    // Get add input feature map
    Value add_ifm = ToBuiltinTensorTypeCast(rewriter, operands[7]);

    // Change appropriate operation over here
    Value conv2dReluVal = rewriter
                               .create<linalg::Conv2DTensorAddReluOp>(
                                   loc, initTensor.getType(),
                                   ValueRange{input, add_ifm, weight, bias}, // add_ifm should be in ValueRange
                                   initTensor, stridesAttr, dilationAttr)
                               .getResult(0);

    if (op->hasAttr("layer_name")) {
      auto attrVal = op->getAttr("layer_name").cast<StringAttr>();
      Operation *conv2dReluValOps = conv2dReluVal.getDefiningOp();
      conv2dReluValOps->setAttr(llvm::StringRef("layer_name"), attrVal);
    }

    auto torchTensorCast = ToTorchTensorTypeCast(rewriter, conv2dReluVal,
                                                 op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dTensorAddLReLUOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dTensorAddLReLUOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dTensorAddLReLUOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2dLRelu = cast<Conv2dTensorAddLReLUOp>(op);
    auto loc = conv2dLRelu.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    if (!isa<Torch::ConstantFloatOp>(operands[7].getDefiningOp()))
      return op->emitError("Alpha, unimplemented: non-floating point type");

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    LogicalResult result = processConv2d(conv2dLRelu, loc, input, elementType, 
        op, rewriter);
    if (result.failed())
      return result;

    // Getting alpha value
    auto c = cast<Torch::ConstantFloatOp>(operands[7].getDefiningOp()).value();
    auto ty = rewriter.getF32Type();
    auto add_const = rewriter.getFloatAttr(ty, c.convertToDouble());
    Value alpha = rewriter.create<arith::ConstantOp>(loc, ty, add_const);

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2dLRelu.stride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2dLRelu.dilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultTensorType.getShape(), elementType);

    // Get add input feature map
    Value add_ifm = ToBuiltinTensorTypeCast(rewriter, operands[8]);

    // Change appropriate operation over here
    Value conv2dLReluVal = rewriter
                               .create<linalg::Conv2DTensorAddLreluOp>(
                                   loc, initTensor.getType(),
                                   ValueRange{input, add_ifm, weight, bias, alpha}, // add_ifm should be in ValueRange
                                   initTensor, stridesAttr, dilationAttr)
                               .getResult(0);

    if (op->hasAttr("layer_name")) {
      auto attrVal = op->getAttr("layer_name").cast<StringAttr>();
      Operation *conv2dLReluValOps = conv2dLReluVal.getDefiningOp();
      conv2dLReluValOps->setAttr(llvm::StringRef("layer_name"), attrVal);
    }

    auto torchTensorCast = ToTorchTensorTypeCast(rewriter, conv2dLReluVal,
                                                 op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dLeakyReluMaxPoolOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dLeakyReluMaxPoolOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dLReLUMaxPoolOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2dLReluMaxpool = cast<Conv2dLReLUMaxPoolOp>(op);
    auto loc = conv2dLReluMaxpool.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    if (!isa<Torch::ConstantFloatOp>(operands[7].getDefiningOp()))
      return op->emitError("Alpha, unimplemented: non-floating point type");

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op->emitError("unimplemented: non-floating point type");

    SmallVector<int64_t> paddingInts;
    paddingInts.resize(2, 0);
    if (!matchPattern(conv2dLReluMaxpool.padding(),
                      Torch::m_TorchConstantIntList(paddingInts))) {
      return rewriter.notifyMatchFailure(
          op, "only support constant padding values");
    }

    // Getting alpha value
    auto c = cast<Torch::ConstantFloatOp>(operands[7].getDefiningOp()).value();
    auto ty = rewriter.getF32Type();
    auto add_const = rewriter.getFloatAttr(ty, c.convertToDouble());
    Value alpha = rewriter.create<arith::ConstantOp>(loc, ty, add_const);

    // paddedInput. input shape change based on padding
    Attribute zeroAttr = rewriter.getZeroAttr(elementType);
    input = applyPad(loc, input, paddingInts, zeroAttr, rewriter);

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2dLReluMaxpool.stride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2dLReluMaxpool.dilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    SmallVector<int64_t, 2> mp_kernel_sizeInts;
    if (!matchPattern(conv2dLReluMaxpool.mp_kernel_size(),
                      Torch::m_TorchConstantIntList(mp_kernel_sizeInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_kernel_size");

    SmallVector<int64_t, 2> mp_strideInts;
    if (!matchPattern(conv2dLReluMaxpool.mp_stride(),
                      Torch::m_TorchConstantIntList(mp_strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int mp_stride");

    SmallVector<int64_t, 2> mp_paddingInts;
    if (!matchPattern(conv2dLReluMaxpool.mp_padding(),
                      Torch::m_TorchConstantIntList(mp_paddingInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_padding");

    SmallVector<int64_t, 2> mp_dilationInts;
    if (!matchPattern(conv2dLReluMaxpool.mp_dilation(),
                      Torch::m_TorchConstantIntList(mp_dilationInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_dilation");

    int64_t groups;
    if (!matchPattern(conv2dLReluMaxpool.groups(),
                      Torch::m_TorchConstantInt(&groups)))
      return rewriter.notifyMatchFailure(op, "only support constant int group");

    bool mp_ceil_mode;
    if (!matchPattern(conv2dLReluMaxpool.mp_ceil_mode(),
                      Torch::m_TorchConstantBool(&mp_ceil_mode)))
      return rewriter.notifyMatchFailure(op,
                                         "only support bool type mp_ceil_mode");

    if (groups != 1)
      return op->emitError("Only support groups value '1'");

    if (mp_ceil_mode)
      return op->emitError("Only support mp_ceil_mode value 'False'");

    SmallVector<int64_t> new_mp_paddingInts; // Hl, Hh, Wl, Wh
    for (uint64_t i = 0; i < mp_paddingInts.size(); i++) {
      new_mp_paddingInts.push_back(mp_paddingInts[i]);
      new_mp_paddingInts.push_back(mp_paddingInts[i]);
    }

    long new_mp_paddingInts_size = new_mp_paddingInts.size();
    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);
    auto mp_kernel_sizeAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_kernel_sizeInts);
    auto mp_stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_strideInts);
    auto mp_paddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({new_mp_paddingInts_size}, rewriter.getI64Type()),
        new_mp_paddingInts);
    auto mp_dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultTensorType.getShape(), elementType);
    auto smallestFPValueAttr = rewriter.getFloatAttr(
        elementType,
        APFloat::getLargest(
            elementType.cast<mlir::FloatType>().getFloatSemantics(),
            /*Negative=*/true));
    Value initValue = rewriter.create<arith::ConstantOp>(loc, smallestFPValueAttr);
    Value filledInitTensor = rewriter.create<linalg::FillOp>(loc, initValue, initTensor).getResult(0);

    Value conv2dLReluMaxpoolVal =
        rewriter
            .create<linalg::Conv2DLreluMaxpoolOp>(
                loc, initTensor.getType(),
                ValueRange{input, weight, bias, alpha}, filledInitTensor, stridesAttr,
                dilationAttr, mp_kernel_sizeAttr, mp_stridesAttr,
                mp_paddingAttr, mp_dilationAttr)
            .getResult(0);

    if (op->hasAttr("layer_name")) {
      auto attrVal = op->getAttr("layer_name").cast<StringAttr>();
      Operation *conv2dReluMaxpoolOps = conv2dLReluMaxpoolVal.getDefiningOp();
      conv2dReluMaxpoolOps->setAttr(llvm::StringRef("layer_name"), attrVal);
    }

    auto torchTensorCast = ToTorchTensorTypeCast(
        rewriter, conv2dLReluMaxpoolVal, op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenConv2dLeakyReluPadMaxPoolOpConversion : public ConversionPattern {
public:
  explicit XTenConv2dLeakyReluPadMaxPoolOpConversion(MLIRContext *context)
      : ConversionPattern(Conv2dLReLUPadMaxPoolOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto conv2dLReluPadMaxpool = cast<Conv2dLReLUPadMaxPoolOp>(op);
    auto loc = conv2dLReluPadMaxpool.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    Value weight = ToBuiltinTensorTypeCast(rewriter, operands[1]);
    Value bias = convertBias(op, operands[2], loc, rewriter);

    if (!isa<Torch::ConstantFloatOp>(operands[7].getDefiningOp()))
      return op->emitError("Alpha, unimplemented: non-floating point type");

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op->emitError("unimplemented: non-floating point type");

    SmallVector<int64_t> paddingInts;
    paddingInts.resize(2, 0);
    if (!matchPattern(conv2dLReluPadMaxpool.padding(),
                      Torch::m_TorchConstantIntList(paddingInts))) {
      return rewriter.notifyMatchFailure(
          op, "only support constant padding values");
    }

    // Getting alpha value
    auto c = cast<Torch::ConstantFloatOp>(operands[7].getDefiningOp()).value();
    auto ty = rewriter.getF32Type();
    auto add_const = rewriter.getFloatAttr(ty, c.convertToDouble());
    Value alpha = rewriter.create<arith::ConstantOp>(loc, ty, add_const);

    // paddedInput. input shape change based on padding
    Attribute zeroAttr = rewriter.getZeroAttr(elementType);
    input = applyPad(loc, input, paddingInts, zeroAttr, rewriter);

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(conv2dLReluPadMaxpool.stride(),
                      Torch::m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");

    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(conv2dLReluPadMaxpool.dilation(),
                      Torch::m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    SmallVector<int64_t> pad_paddingInts; // Wl, Wh, Hl, Hh
    if (!matchPattern(conv2dLReluPadMaxpool.pad_padding(),
                      Torch::m_TorchConstantIntList(pad_paddingInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int pad_padding");

    SmallVector<int64_t, 2> mp_kernel_sizeInts;
    if (!matchPattern(conv2dLReluPadMaxpool.mp_kernel_size(),
                      Torch::m_TorchConstantIntList(mp_kernel_sizeInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_kernel_size");

    SmallVector<int64_t, 2> mp_strideInts;
    if (!matchPattern(conv2dLReluPadMaxpool.mp_stride(),
                      Torch::m_TorchConstantIntList(mp_strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int mp_stride");

    SmallVector<int64_t> mp_paddingInts; // H, W
    if (!matchPattern(conv2dLReluPadMaxpool.mp_padding(),
                      Torch::m_TorchConstantIntList(mp_paddingInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_padding");

    SmallVector<int64_t, 2> mp_dilationInts;
    if (!matchPattern(conv2dLReluPadMaxpool.mp_dilation(),
                      Torch::m_TorchConstantIntList(mp_dilationInts)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int mp_dilation");

    int64_t groups;
    if (!matchPattern(conv2dLReluPadMaxpool.groups(),
                      Torch::m_TorchConstantInt(&groups)))
      return rewriter.notifyMatchFailure(op, "only support constant int group");

    bool mp_ceil_mode;
    if (!matchPattern(conv2dLReluPadMaxpool.mp_ceil_mode(),
                      Torch::m_TorchConstantBool(&mp_ceil_mode)))
      return rewriter.notifyMatchFailure(op,
                                         "only support bool type mp_ceil_mode");

    if (groups != 1)
      return op->emitError("Only support groups value '1'");

    if (mp_ceil_mode)
      return op->emitError("Only support mp_ceil_mode value 'False'");

    SmallVector<int64_t> pad_mp_paddingInts; // Hl,Hh, Wl, Wh
    if (pad_paddingInts.size() != mp_paddingInts.size() * 2)
      return rewriter.notifyMatchFailure(
          op, "max_pool padding is not double of pad padding");

    for (uint64_t i = mp_paddingInts.size(); i > 0; --i) {
      pad_mp_paddingInts.push_back(pad_paddingInts[i * 2 - 2] +
                                   mp_paddingInts[i - 1]);
      pad_mp_paddingInts.push_back(pad_paddingInts[i * 2 - 1] +
                                   mp_paddingInts[i - 1]);
    }

    long pad_mp_padding_size = pad_mp_paddingInts.size();

    auto stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), strideInts);
    auto dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), dilationInts);
    auto mp_kernel_sizeAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_kernel_sizeInts);
    auto mp_stridesAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_strideInts);
    auto pad_mp_paddingAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({pad_mp_padding_size}, rewriter.getI64Type()),
        pad_mp_paddingInts);
    auto mp_dilationAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({2}, rewriter.getI64Type()), mp_dilationInts);

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultTensorType.getShape(), elementType);
    auto smallestFPValueAttr = rewriter.getFloatAttr(
        elementType,
        APFloat::getLargest(
            elementType.cast<mlir::FloatType>().getFloatSemantics(),
            /*Negative=*/true));
    Value initValue = rewriter.create<arith::ConstantOp>(loc, smallestFPValueAttr);
    Value filledInitTensor = rewriter.create<linalg::FillOp>(loc, initValue, initTensor).getResult(0);

    Value conv2dLReluPadMaxpoolVal =
        rewriter
            .create<linalg::Conv2DLreluMaxpoolOp>(
                loc, initTensor.getType(),
                ValueRange{input, weight, bias, alpha}, filledInitTensor, stridesAttr,
                dilationAttr, mp_kernel_sizeAttr, mp_stridesAttr,
                pad_mp_paddingAttr, mp_dilationAttr)
            .getResult(0);

    if (op->hasAttr("layer_name")) {
      auto attrVal = op->getAttr("layer_name").cast<StringAttr>();
      Operation *conv2dLReluPadMaxpoolOps =
          conv2dLReluPadMaxpoolVal.getDefiningOp();
      conv2dLReluPadMaxpoolOps->setAttr(llvm::StringRef("layer_name"), attrVal);
    }

    auto torchTensorCast = ToTorchTensorTypeCast(
        rewriter, conv2dLReluPadMaxpoolVal, op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenPartialConv2dReLUOpConversion : public ConversionPattern {
public:
  explicit XTenPartialConv2dReLUOpConversion(MLIRContext *context)
      : ConversionPattern(PartialConv2dReLUOp::getOperationName(), 1, context) {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    /*
     auto mmult = cast<PartialConv2dReLUOp>(op);
     auto loc = mmult.getLoc();

     auto A = MemRefTypeCast(rewriter, operands[0]);
     auto B = MemRefTypeCast(rewriter, mmult.weight());

     auto resultTy = op->getResult(0).getType();
     auto tensorResultTy = resultTy.cast<Torch::BaseTensorType>();
     auto memRefResultTy = mlir::MemRefType::get(
         tensorResultTy.getSizes(), tensorResultTy.getDtype(), {}, 0);

     Value C;
     if(mmult.PartialIn()) {
       C = mmult.PartialIn();
     } else {
       C = rewriter.create<memref::AllocOp>(loc, memRefResultTy).getResult();
     }

     rewriter.create<linalg::Conv2DNhwcHwcfOp>(loc, ValueRange{A, B},
     ValueRange{C});

     auto tensor_cast = TensorTypeCast(rewriter, C, op->getResult(0).getType());

     if(mmult.getNumResults() == 1)
       rewriter.replaceOp(op, tensor_cast);
     else
       rewriter.replaceOp(op, {tensor_cast, operands[0]});
     */
    return failure();
  }
};

class XTenSoftmaxOpConversion : public ConversionPattern {
public:
  explicit XTenSoftmaxOpConversion(MLIRContext *context)
      : ConversionPattern(SoftmaxOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto softmax = cast<SoftmaxOp>(op);
    auto loc = softmax.getLoc();

    Value input = ToBuiltinTensorTypeCast(rewriter, operands[0]);
    auto torchDim = (operands[1].getDefiningOp<Torch::ConstantIntOp>()).value();
    auto dim = rewriter.getI64IntegerAttr(torchDim.getSExtValue());

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();

    auto torchTensorTy =
        op->getResult(0).getType().cast<Torch::BaseTensorType>();
    auto resultTensorType = RankedTensorType::get(torchTensorTy.getSizes(),
                                                  torchTensorTy.getDtype());

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultTensorType.getShape(), elementType);

    Value softmaxVal =
        rewriter.create<linalg::SoftmaxOp>(loc, initTensor.getType(), input, dim)
            .getResult();

    auto torchTensorCast =
        ToTorchTensorTypeCast(rewriter, softmaxVal, op->getResult(0).getType());
    rewriter.replaceOp(op, torchTensorCast);
    return success();
  }
};

class XTenToLinalgPass : public XTenToLinalgBase<XTenToLinalgPass> {

public:
  XTenToLinalgPass() = default;
  XTenToLinalgPass(const XTenToLinalgPass &pass) {};

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();

    TypeConverter typeConverter;

    // tablegen patterns
    RewritePatternSet patterns(context);

    patterns.insert<XTenAddOpConversion,
                    XTenMulOpConversion,
                    XTenMMOpConversion,
                    XTenConv2dOpConversion,
                    XTenConv2dReluOpConversion,
                    XTenConv2dLeakyReluOpConversion,
                    XTenConv2dLeakyReluMaxPoolOpConversion,
                    XTenConv2dLeakyReluPadMaxPoolOpConversion,
                    XTenPartialConv2dReLUOpConversion,
                    XTenConv2dTensorAddOpConversion,
                    XTenConv2dTensorAddReLUOpConversion,
                    XTenConv2dTensorAddLReLUOpConversion,
                    XTenSoftmaxOpConversion>(context);

    ConversionTarget target(*context);
    
    target.addIllegalDialect<XTenDialect>();
    target.addLegalDialect<
        linalg::LinalgDialect,
        arith::ArithmeticDialect, scf::SCFDialect, tensor::TensorDialect, Torch::TorchDialect, 
        TorchConversion::TorchConversionDialect>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering XTen to Linalg\n");
      signalPassFailure();
    }
  }

private:
};

} // namespace

namespace xilinx {
namespace xten {

std::unique_ptr<Pass> createXTenToLinalgPass() {
  return std::make_unique<XTenToLinalgPass>();
}

} // namespace xten
} // namespace xilinx
