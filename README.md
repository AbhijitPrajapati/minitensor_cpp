# MiniTensor

MiniTensor is a lazily evaluated machine learning library written in C++. Currently a work in progress, it features tensor operations with broadcasting and non-contiguous layouts, reverse-mode automatic differentiation, explicit random-key management, typed parameter trees, basic module composition, loss functions, and optimizers, all executed on the CPU.

Rather than running immediately, kernels are executed only when tensor data is requested. Similarly, automatic differentiation constructs a separate lazy graph instead of evaluating outputs or targets. 

Neural network modules act as immutable computation specifications, while their parameters are externally managed by the user. The same goes for optimizer state.

See [`demo.cpp`](demo.cpp) for a complete training demonstration of a composite dense model on a three-class spiral dataset.

## Building

MiniTensor requires a C++20 compiler, CMake 3.24 or newer, and Ninja. The included CMake presets support GCC, Clang, and MSVC.

```powershell
cmake --preset gcc-release
cmake --build --preset gcc-release
ctest --preset gcc-release
```

The training demonstration can then be run with:

```powershell
.\build\gcc-release\minitensor_demo.exe
```

GoogleTest is downloaded by CMake when the test suite is configured.

## Current Scope

MiniTensor currently supports a CPU backend, and provides `Float32` and `Int64` tensors. Operations required for basic training have been implemented, including tensor creation, data transfer, elementwise operations, reductions, matrix multiplication, indexing, and concetenation. Only linear modules are provided at the moment, along with a rudimentary set of loss functions and optimizers.

## Future Development

MiniTensor will expand to accommodate a CUDA backend as well as a more optimized CPU backend. It will also take further advantage of lazy evaluation through reusable, compiled computation graphs. Convenient checkpointing and model persistence will be added to support more extensive training, and the operation set and selection of datatypes will continue to grow.