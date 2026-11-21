#include "tensor/backend/cuda/kernels/registrations.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <numbers>
#include <span>
#include <type_traits>

#include "tensor/backend/cuda/detail/buffer_access.hpp"
#include "tensor/backend/cuda/detail/cuda_check.hpp"
#include "tensor/backend/cuda/detail/device_guard.hpp"
#include "tensor/backend/device_runtime.hpp"
#include "tensor/backend/dtype_dispatch.hpp"
#include "tensor/backend/random_engine.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/primitives/creation/full.hpp"
#include "tensor/primitives/creation/normal.hpp"
#include "tensor/primitives/creation/uniform.hpp"
#include "tensor/random_key_access.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cuda
{
namespace
{
template <TensorElement T>
__global__ void full_kernel(T *output, std::size_t count, T value)
{
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = start; index < count; index += step)
    {
        output[index] = value;
    }
}

void run_full(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
              MutableTensorView output)
{
    assert(inputs.empty());
    (void)inputs;
    assert(runtime.device() == output.device());
    (void)runtime;
    assert(output.layout().is_contiguous(output.shape()));

    const auto &full = dynamic_cast<const FullPrimitive &>(primitive);
    const std::size_t count = output.shape().numel();
    if (count == 0)
    {
        return;
    }

    constexpr unsigned int threads = 256;
    constexpr std::size_t max_blocks = 65535;
    const auto blocks = static_cast<unsigned int>(std::min(count / threads + (count % threads != 0), max_blocks));

    DeviceGuard guard(output.device());

    dispatch_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        T *destination = data<T>(output) + output.layout().offset();
        const T value = full.fill_value().get<T>();
        full_kernel<T><<<blocks, threads>>>(destination, count, value);
        cuda_check(cudaGetLastError(), "launch full kernel");
    });
    cuda_check(cudaStreamSynchronize(nullptr), "complete full kernel");
    guard.restore();
}

template <TensorElement T>
__global__ void uniform_kernel(T *output, std::size_t count, T low, T high, T largest_valid, std::uint64_t key_first,
                               std::uint64_t key_second)
{
    static_assert(std::is_floating_point_v<T>);

    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const std::size_t random_block_count = count / RandomBlock::word_count + (count % RandomBlock::word_count != 0);
    const double low_double = static_cast<double>(low);
    const double range = static_cast<double>(high) - low_double;

    for (std::size_t block_index = start; block_index < random_block_count; block_index += step)
    {
        const RandomBlock block = random_block(key_first, key_second, block_index);
        const std::size_t base = block_index * RandomBlock::word_count;
        for (std::size_t lane = 0; lane < RandomBlock::word_count && lane < count - base; ++lane)
        {
            const double unit = static_cast<double>(unit_uniform(block.words[lane]));
            T value = static_cast<T>(low_double + range * unit);
            // Rounding to the output dtype must not include the upper bound.
            if (value >= high)
            {
                value = largest_valid;
            }
            output[base + lane] = value;
        }
    }
}

void run_uniform(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                 MutableTensorView output)
{
    assert(inputs.empty());
    (void)inputs;
    assert(runtime.device() == output.device());
    (void)runtime;
    assert(output.layout().is_contiguous(output.shape()));

    const auto &uniform = dynamic_cast<const UniformPrimitive &>(primitive);
    const std::size_t count = output.shape().numel();
    if (count == 0)
    {
        return;
    }

    constexpr unsigned int threads = 256;
    constexpr std::size_t max_blocks = 65535;
    DeviceGuard guard(output.device());

    dispatch_floating_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        T *destination = data<T>(output) + output.layout().offset();
        const auto low = static_cast<T>(uniform.low());
        const auto high = static_cast<T>(uniform.high());
        const std::size_t work_count =
            low == high ? count : count / RandomBlock::word_count + (count % RandomBlock::word_count != 0);
        const auto blocks =
            static_cast<unsigned int>(std::min(work_count / threads + (work_count % threads != 0), max_blocks));

        if (low == high)
        {
            // Use full if low == high.
            full_kernel<T><<<blocks, threads>>>(destination, count, low);
        }
        else
        {
            const auto [first, second] = RandomKeyAccess::words(uniform.key());
            const T largest_valid = std::nextafter(high, low);
            uniform_kernel<T><<<blocks, threads>>>(destination, count, low, high, largest_valid, first, second);
        }
        cuda_check(cudaGetLastError(), "launch uniform kernel");
    });
    cuda_check(cudaStreamSynchronize(nullptr), "complete uniform kernel");
    guard.restore();
}

template <TensorElement T>
__global__ void normal_kernel(T *output, std::size_t count, T mean, T std_dev, std::uint64_t key_first,
                              std::uint64_t key_second)
{
    static_assert(std::is_floating_point_v<T>);

    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const std::size_t random_block_count = count / RandomBlock::word_count + (count % RandomBlock::word_count != 0);
    const auto mean_double = static_cast<double>(mean);
    const auto std_dev_double = static_cast<double>(std_dev);
    constexpr double two_pi = 2.0 * std::numbers::pi_v<double>;

    for (std::size_t block_index = start; block_index < random_block_count; block_index += step)
    {
        const RandomBlock block = random_block(key_first, key_second, block_index);
        const std::size_t base = block_index * RandomBlock::word_count;
        for (std::size_t lane = 0; lane < RandomBlock::word_count && lane < count - base; lane += 2)
        {
            const double u1 = open_unit_uniform(block.words[lane]);
            const double u2 = open_unit_uniform(block.words[lane + 1]);
            const double radius = ::sqrt(-2.0 * ::log(u1));
            const double angle = two_pi * u2;
            double sine;
            double cosine;
            ::sincos(angle, &sine, &cosine);

            output[base + lane] = static_cast<T>(mean_double + std_dev_double * (radius * cosine));
            if (lane + 1 < count - base)
            {
                output[base + lane + 1] = static_cast<T>(mean_double + std_dev_double * (radius * sine));
            }
        }
    }
}

void run_normal(DeviceRuntime &runtime, const Primitive &primitive, std::span<const TensorView> inputs,
                MutableTensorView output)
{
    assert(inputs.empty());
    (void)inputs;
    assert(runtime.device() == output.device());
    (void)runtime;
    assert(output.layout().is_contiguous(output.shape()));

    const auto &normal = dynamic_cast<const NormalPrimitive &>(primitive);
    const std::size_t count = output.shape().numel();
    if (count == 0)
    {
        return;
    }

    constexpr unsigned int threads = 256;
    constexpr std::size_t max_blocks = 65535;
    DeviceGuard guard(output.device());

    dispatch_floating_dtype(output.dtype(), [&]<typename T>(std::type_identity<T>) {
        T *destination = data<T>(output) + output.layout().offset();
        const auto mean = static_cast<T>(normal.mean());
        const auto std_dev = static_cast<T>(normal.std_dev());
        const std::size_t work_count =
            std_dev == T{0} ? count : count / RandomBlock::word_count + (count % RandomBlock::word_count != 0);
        const auto blocks =
            static_cast<unsigned int>(std::min(work_count / threads + (work_count % threads != 0), max_blocks));

        if (std_dev == T{0})
        {
            // Use full if standard deviation is 0.
            full_kernel<T><<<blocks, threads>>>(destination, count, mean);
        }
        else
        {
            const auto [first, second] = RandomKeyAccess::words(normal.key());
            normal_kernel<T><<<blocks, threads>>>(destination, count, mean, std_dev, first, second);
        }
        cuda_check(cudaGetLastError(), "launch normal kernel");
    });
    cuda_check(cudaStreamSynchronize(nullptr), "complete normal kernel");
    guard.restore();
}

} // namespace

void register_creation_kernels(KernelRegistry &registry)
{
    registry.register_kernel(KernelKey{FullPrimitive::ID, DeviceType::Cuda}, run_full);
    registry.register_kernel(KernelKey{UniformPrimitive::ID, DeviceType::Cuda}, run_uniform);
    registry.register_kernel(KernelKey{NormalPrimitive::ID, DeviceType::Cuda}, run_normal);
}
} // namespace minitensor::detail::cuda
