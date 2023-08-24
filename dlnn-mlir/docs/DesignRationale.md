# Design Rationale

This document attempts to explain the design rationale behind the `dlnn` dialect, and why we thought we had to make it.

## Motivation

Within the **MLIR** ecosystem, there are already a multitude of dialects that are closely related to modelling deep learning neural networks. Let's roughly divide them into 3 categories:

- Ingestion

  Dialects like [`onnx-mlir`][onnx-mlir] and [`torch-mlir`][torch-mlir] are primarily created for the purpose of ingesting a description of a concrete neural network into an **MLIR**-based workflow.

- Transformation

  [`flow`][flow] is a dialect within [**IREE**][iree] that is used to encode and transform the dispatching of tensor operations in a neural network, represented by a graph.

- Code generation

  Although not limited to DL, the [`linalg`][linalg] dialect exists to capture operations on shaped types, such as tensors, within the linear algebra domain. It has growing support for transformations, but historically it offers a bridge from tensorland to (affine) loop nests operating on memory.

Another particularly interesting example is the [`tosa`][tosa] dialect, which is modeled after the eponymous [specification](https://developer.mlplatform.org/w/tosa/). It promises interoperability by defining a set of operators with very constrained semantics, which provide an ISA between model and hardware developers. As such, it is valuable for *ingesting* interoperable networks, but also useful during *code generation* for hardware that implements the spec.

### `dlnn`

The `dlnn` dialect aims to straddle somwhere between transformations and code generation. Here's why we needed a new dialect for this:

- Front-end specific dialects usually have limited flexibility for data types, and generally can't encode target-specific decisions, such as memory layout or loop schedule.
- Existing transformation dialects are often proprietary to their user projects, and no existing ones were designed to deal with the flexibility we needed.
- Most dialects suitable for code generation are either (deliberately) too limited in terms of encodable design decisison (such as `tosa`), or too generic to be able to reliably transform on them.

There certainly remains a gap towards code generation, where a slightly more constrained tensor-level operator abstraction could live. See, for example, **IREE**'s extensions to `linalg` sometimes called `linalg-ex`.

For this project, we wanted to focus on a simpler encoding, which is not so much a general tensor abstraction, but rather a very domain-constrained deep learning one.

## Design

We wanted a dialect that would...

- ...capture the deep learning domain (nothing more, nothing less).
- ...have as much implicit but common sense semantics as possible.
- ...have a simple lowering / path towards implementations.
- ...allow for making incremental lowering decisions / pinning details.

In particular, we wanted to make sure that similar network operators would be encoded with the same MLIR operation for as long as possible.

The mechanism that allows this are the custom types `!dlnn.fm` and `!dlnn.weights`, which this section will describe in detail.

### `!dlnn.fm`

The feature map type is a value tensor abstraction with a logical structure and an optional link to a physical (memory) one. A feature map has the following properties:

- Scalar type
- Channels

  An extent (dimension size) that indicates logical channels of feature data.

- Sizes

  A series of extents (dimension sizes) that are logical sizes of the feature volume.

- Layout map (optional)

  An affine mapping that projects from logical indices to a concrete memory layout.

For example, the feature map type `!dlnn.fm<3xf32[128,128], (W,H,C)[N] -> (N,C,H,W)>` tells us that:

- The scalars are of type `f32`.
- There are 3 channels.
- The volume is 128 by 128.
- The chosen tensor layout is `NCHW`.

In other words, this trivially lowers to a `tensor<?x3x128x128xf32>`.

### `!dlnn.weights`

The weight type is also a value tensor abstraction, but its logical structure is geared towards it being used as part of a transfer function. It has the following properties:

- Input channels

  An extent (dimension size) that indicates logical channels of input data.

- Output channels

  An extent (dimension size) that indicates logical channels of output data.

- Sizes

  A series of extents (dimension sizes) that are logical sizes of the filter kernel volume, if applicable.

An example of a weight type that is applicable to our previous feature map type would be `!dlnn.weights<3->3x3xf32->16>`. Weights for a fully-connected layer for the same feature map type would be `!dlnn.weights<3->128x128xf32->3>`.

---

[mlir]: https://mlir.llvm.org
[onnx-mlir]: https://github.com/onnx/onnx-mlir
[torch-mlir]: https://github.com/llvm/torch-mlir
[flow]: https://github.com/google/iree/tree/main/compiler/src/iree/compiler/Dialect/Flow
[iree]: https://google.github.io/iree
[linalg]: https://mlir.llvm.org/docs/Dialects/Linalg/
[tosa]: https://mlir.llvm.org/docs/Dialects/TOSA/
