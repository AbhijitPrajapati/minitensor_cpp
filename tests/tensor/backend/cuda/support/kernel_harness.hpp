#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/backend/cuda/cuda_runtime.hpp"
#include "tensor/backend/cuda/register_kernels.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/storage/buffer.hpp"

namespace minitensor::test
{
class CudaKernelHarness final
{
  public:
    CudaKernelHarness()
    {
        detail::cuda::register_kernels(registry_);
    }

    template <TensorElement T>
    [[nodiscard]] detail::BufferRef allocate(std::size_t element_count)
    {
        return runtime_.allocate(element_count * sizeof(T));
    }

    template <TensorElement T>
    void copy_from_host(const detail::BufferRef &buffer, std::span<const T> values)
    {
        runtime_.copy_from_host(*buffer, 0, std::as_bytes(values));
    }

    template <TensorElement T>
    [[nodiscard]] std::vector<T> copy_to_host(const detail::BufferRef &buffer)
    {
        std::vector<T> result(buffer->size_bytes() / sizeof(T));
        runtime_.copy_to_host(std::as_writable_bytes(std::span<T>(result)), *buffer, 0);
        return result;
    }

    void invoke(const detail::Primitive &primitive, std::span<const detail::TensorView> inputs,
                detail::MutableTensorView output)
    {
        const detail::KernelFn &kernel = registry_.get({primitive.id(), DeviceType::Cuda});
        kernel(runtime_, primitive, inputs, output);
    }

    [[nodiscard]] detail::cuda::CudaRuntime &runtime() noexcept
    {
        return runtime_;
    }

  private:
    detail::cuda::CudaRuntime runtime_;
    detail::KernelRegistry registry_;
};
} // namespace minitensor::test
