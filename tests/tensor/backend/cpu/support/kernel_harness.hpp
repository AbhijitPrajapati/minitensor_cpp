#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/cpu_buffer.hpp"
#include "tensor/backend/cpu/cpu_runtime.hpp"
#include "tensor/backend/cpu/register_kernels.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/storage/buffer.hpp"

namespace minitensor::test
{
class CpuKernelHarness final
{
  public:
    CpuKernelHarness()
    {
        detail::cpu::register_kernels(registry_);
    }

    template <TensorElement T>
    [[nodiscard]] detail::BufferRef allocate(std::size_t element_count)
    {
        return runtime_.allocate(element_count * sizeof(T));
    }

    template <TensorElement T>
    [[nodiscard]] T *data(const detail::BufferRef &buffer) const
    {
        auto *cpu_buffer = dynamic_cast<detail::cpu::CpuBuffer *>(buffer.get());
        if (cpu_buffer == nullptr)
        {
            throw std::logic_error{"CPU kernel test requires CPU storage"};
        }
        return reinterpret_cast<T *>(cpu_buffer->data());
    }

    void invoke(const detail::Primitive &primitive, std::span<const detail::TensorView> inputs,
                detail::MutableTensorView output)
    {
        const detail::KernelFn &kernel = registry_.get({primitive.id(), DeviceType::Cpu});
        kernel(runtime_, primitive, inputs, output);
    }

    [[nodiscard]] detail::cpu::CpuRuntime &runtime() noexcept
    {
        return runtime_;
    }

  private:
    detail::KernelRegistry registry_;
    detail::cpu::CpuRuntime runtime_;
};
} // namespace minitensor::test
