# Deep Learning Neural Network (`dlnn`) Dialect

The `dlnn-mlir` project implements an [**MLIR**][mlir] dialect `dlnn` that provides a flexible abstraction for primitives in deep learning neural networks.

## Building

The `dlnn-mlir` project is built using **CMake** (version `3.20` or newer). Make sure to provide all dependencies required by the project, either by installing them to system-default locations, or by setting the appropriate search location hints!

```sh
# Configure.
cmake -S . -B build \
    -G Ninja \
    -DLLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm \
    -DMLIR_DIR=$MLIR_PREFIX/lib/cmake/mlir

# Build.
cmake --build build
```

The following CMake variables can be configured:

|       Name | Type     | Description                                                                                            |
| ---------: | :------- | ------------------------------------------------------------------------------------------------------ |
| `LLVM_DIR` | `STRING` | Path to the CMake directory of an **LLVM** installation. <br/> *e.g. `~/tools/llvm-15/lib/cmake/llvm`* |
| `MLIR_DIR` | `STRING` | Path to the CMake directory of an **MLIR** installation. <br/> *e.g. `~/tools/llvm-15/lib/cmake/mlir`* |

## License

> TODO

---

[mlir]: https://mlir.llvm.org
[onnx-mlir]: https://github.com/onnx/onnx-mlir
[torch-mlir]: https://github.com/llvm/torch-mlir
[flow]: https://github.com/google/iree/tree/main/compiler/src/iree/compiler/Dialect/Flow
[iree]: https://google.github.io/iree
[linalg]: https://mlir.llvm.org/docs/Dialects/Linalg/
[tosa]: https://mlir.llvm.org/docs/Dialects/TOSA/
