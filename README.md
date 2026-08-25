# minitensor

`minitensor` is a deliberately small CPU/f32 tensor library for studying how
tensor storage, strided views, broadcasting, numerical kernels, and reverse-mode
automatic differentiation fit together.

The project favors explicit scalar implementations over performance machinery.
There is no dtype dispatcher, device abstraction, public in-place mutation, or
hidden conversion to contiguous storage in ordinary arithmetic kernels.

## Example

```cpp
#include "minitensor/minitensor.hpp"

using minitensor::Shape;
using minitensor::Tensor;

auto x = Tensor::from_data({1, 2, 3, 4, 5, 6}, Shape{2, 3}, true);
auto weights = Tensor::from_data({0.5F, 1.0F, 1.5F}, Shape{3});

auto loss = minitensor::relu(x * weights).sum();
loss.backward();

auto dx = x.grad()->to_vector();
```

## Architecture

A public `Tensor` is a cheap shared handle to an internal `TensorImpl`.
`TensorImpl` owns a layout by value and shares a `Storage` allocation:

```text
Tensor
  -> shared TensorImpl
       -> shared Storage (one contiguous std::vector<float>)
       -> Layout (shape, positive strides, element offset)
       -> AutogradMeta
            -> optional accumulated leaf gradient
            -> optional backward Node -> parent Tensor handles
```

Copying a `Tensor` copies its handle. Transpose and slice create a new tensor
identity and layout while sharing storage. Numerical operations allocate new,
contiguous storage. `reshape` shares storage only when its input is contiguous;
otherwise it materializes logical element order first.

`Shape` remains a public `std::vector<Index>` alias for simple construction.
Private shape-geometry functions validate extents and centralize checked element
counts, broadcasting, reduction-shape inference, reshape compatibility, and
logical coordinate conversion. `Layout` builds on that geometry and owns
stride- and offset-aware view transformations such as transpose, slice, reshape,
and internal broadcasting.

The implementation is divided into four boundaries:

- `src/core`: shape algebra, storage, layout invariants, and private tensor state.
- `src/kernels`: non-owning read/write arguments, private iteration/execution
  plans, and scalar strided kernels.
- `src/ops`: public orchestration, output construction, and gradient rules.
- `src/autograd`: graph nodes, reverse-topological traversal, and accumulation.

Kernels do not inspect autograd state. Autograd nodes point only toward parents,
so temporary intermediates remain alive without parent-to-child reference cycles.
Saved values are detached handles, and public data is immutable, so this baseline
does not need mutation version counters.

## Supported operations

- Trailing-dimension broadcasting for `+`, `-`, `*`, and `/`
- Rank-2 matrix multiplication
- Whole-tensor or single-axis `sum`, with optional `keepdim`
- Positive-step, single-axis slicing and dimension transpose
- `contiguous` and `reshape`
- ReLU, sigmoid, and tanh
- Reverse-mode autodiff with leaf gradient accumulation and explicit gradient
  seeds for non-scalar outputs

Rank-zero shape `{}` is a scalar. Shapes containing a zero extent are supported.
Shapes, strides, offsets, and logical indices use signed 64-bit integers. Negative
shapes and indices are rejected; negative indexing and negative strides are not
implemented. Batched matmul, additional dtypes/devices, and public in-place
operations are intentionally outside the current scope.

## Build and test

From a Visual Studio developer shell:

```powershell
cmake --preset x64-debug
cmake --build --preset x64-debug
ctest --preset x64-debug
```

The public API test uses only installed-style headers, which helps catch accidental
leakage of internal interfaces. A separate internal layout test directly verifies
coordinate conversion and bounds validation.
