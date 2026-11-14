#pragma once

#include <cassert>
#include <minitensor/types.hpp>

#include "tensor/backend/cuda/cuda_buffer.hpp"
#include "tensor/dispatch/tensor_view.hpp"

namespace minitensor::detail::cuda
{
// These helper functions access cuda buffers and return pointers.

template <TensorElement T>
[[nodiscard]] const T *data(const TensorView &view)
{
    assert(view.dtype() == dtype_of_v<T>);
    const auto &buffer = dynamic_cast<const CudaBuffer &>(view.buffer());
    return reinterpret_cast<const T *>(buffer.data());
}

template <TensorElement T>
[[nodiscard]] T *data(const MutableTensorView &view)
{
    assert(view.dtype() == dtype_of_v<T>);
    auto &buffer = dynamic_cast<CudaBuffer &>(view.buffer());
    return reinterpret_cast<T *>(buffer.data());
}
} // namespace minitensor::detail::cuda