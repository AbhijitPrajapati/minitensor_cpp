#include <minitensor/types.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <typeinfo>

#include "tensor/backend/cpu/cpu_buffer.hpp"
#include "tensor/backend/cpu/cpu_runtime.hpp"
#include "tensor/backend/cpu/register_kernels.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/ops/add.hpp"
#include "tensor/ops/full.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

#include "../support/test.hpp"

namespace minitensor::test
{
    namespace
    {
        detail::cpu::CpuBuffer &as_cpu_buffer(const detail::BufferRef &buffer)
        {
            expect(buffer != nullptr, "a cpu kernel fixture requires a buffer");
            auto *cpu_buffer = dynamic_cast<detail::cpu::CpuBuffer *>(buffer.get());
            expect(cpu_buffer != nullptr, "a cpu kernel fixture requires cpu storage");
            return *cpu_buffer;
        }
    }

    void run_cpu_kernels_test()
    {
        using detail::AddPrimitive;
        using detail::BufferRef;
        using detail::FullPrimitive;
        using detail::KernelFn;
        using detail::KernelKey;
        using detail::KernelRegistry;
        using detail::Layout;
        using detail::Materialization;
        using detail::MutableTensorView;
        using detail::TensorSpec;
        using detail::TensorView;
        using detail::cpu::CpuRuntime;

        KernelRegistry registry;
        detail::cpu::register_kernels(registry);

        KernelKey full_key{typeid(FullPrimitive), DeviceType::Cpu, DType::Float32};
        KernelKey add_key{typeid(AddPrimitive), DeviceType::Cpu, DType::Float32};
        expect(registry.contains(full_key), "cpu kernel registration installs the Float32 full kernel");
        expect(registry.contains(add_key), "cpu kernel registration installs the Float32 add kernel");

        CpuRuntime runtime;
        const KernelFn &full_kernel = registry.get(full_key);
        const KernelFn &add_kernel = registry.get(add_key);

        const TensorSpec full_spec{Shape{2, 3}, DType::Float32, Device::cpu()};
        const BufferRef full_buffer = runtime.allocate(7 * sizeof(float));
        auto &full_cpu_buffer = as_cpu_buffer(full_buffer);
        auto *full_data = reinterpret_cast<float *>(full_cpu_buffer.data());
        std::fill_n(full_data, 7, -10.0F);

        const Layout full_layout{{3, 1}, 1};
        const Materialization full_materialization{full_buffer, full_layout};
        const MutableTensorView full_output{full_spec, full_materialization};
        const FullPrimitive full_primitive{full_spec, 2.5F};
        full_kernel(runtime, full_primitive, std::span<const TensorView>{}, full_output);

        expect(full_data[0] == -10.0F, "the full kernel respects the output layout offset");
        expect(std::all_of(full_data + 1, full_data + 7, [](float value)
                           { return value == 2.5F; }),
               "the full kernel fills every logical output element");

        const TensorSpec empty_spec{Shape{2, 0}, DType::Float32, Device::cpu()};
        const BufferRef empty_full_buffer = runtime.allocate(0);
        auto &empty_full_cpu_buffer = as_cpu_buffer(empty_full_buffer);
        const Materialization empty_full_materialization{
            empty_full_buffer,
            Layout::contiguous(empty_spec.shape)};
        const MutableTensorView empty_full_output{empty_spec, empty_full_materialization};
        const FullPrimitive empty_full_primitive{empty_spec, 4.0F};
        full_kernel(runtime, empty_full_primitive, std::span<const TensorView>{}, empty_full_output);
        expect(empty_full_cpu_buffer.data() == nullptr, "the full kernel accepts an empty output without storage");

        const TensorSpec lhs_spec{Shape{2, 1}, DType::Float32, Device::cpu()};
        const TensorSpec rhs_spec{Shape{1, 3}, DType::Float32, Device::cpu()};
        const TensorSpec sum_spec{Shape{2, 3}, DType::Float32, Device::cpu()};
        const BufferRef lhs_buffer = runtime.allocate(2 * sizeof(float));
        const BufferRef rhs_buffer = runtime.allocate(3 * sizeof(float));
        const BufferRef sum_buffer = runtime.allocate(6 * sizeof(float));
        auto &lhs_cpu_buffer = as_cpu_buffer(lhs_buffer);
        auto &rhs_cpu_buffer = as_cpu_buffer(rhs_buffer);
        auto &sum_cpu_buffer = as_cpu_buffer(sum_buffer);

        const std::array<float, 2> lhs_values{1.0F, 2.0F};
        const std::array<float, 3> rhs_values{10.0F, 20.0F, 30.0F};
        std::copy(lhs_values.begin(), lhs_values.end(), reinterpret_cast<float *>(lhs_cpu_buffer.data()));
        std::copy(rhs_values.begin(), rhs_values.end(), reinterpret_cast<float *>(rhs_cpu_buffer.data()));

        const Materialization lhs_materialization{lhs_buffer, Layout::contiguous(lhs_spec.shape)};
        const Materialization rhs_materialization{rhs_buffer, Layout::contiguous(rhs_spec.shape)};
        const Materialization sum_materialization{sum_buffer, Layout::contiguous(sum_spec.shape)};
        const std::array<TensorView, 2> add_inputs{
            TensorView{lhs_spec, lhs_materialization},
            TensorView{rhs_spec, rhs_materialization}};
        const MutableTensorView sum_output{sum_spec, sum_materialization};
        const AddPrimitive add_primitive;
        add_kernel(runtime, add_primitive, add_inputs, sum_output);

        const std::array<float, 6> expected_sum{11.0F, 21.0F, 31.0F, 12.0F, 22.0F, 32.0F};
        const auto *sum_data = reinterpret_cast<const float *>(sum_cpu_buffer.data());
        expect(std::equal(expected_sum.begin(), expected_sum.end(), sum_data),
               "the add kernel performs trailing-dimension broadcasting");

        const BufferRef empty_add_buffer = runtime.allocate(0);
        auto &empty_add_cpu_buffer = as_cpu_buffer(empty_add_buffer);
        const Materialization empty_add_materialization{
            empty_add_buffer,
            Layout::contiguous(empty_spec.shape)};
        const TensorView empty_input{empty_spec, empty_add_materialization};
        const std::array<TensorView, 2> empty_inputs{empty_input, empty_input};
        const MutableTensorView empty_sum_output{empty_spec, empty_add_materialization};
        add_kernel(runtime, add_primitive, empty_inputs, empty_sum_output);
        expect(empty_add_cpu_buffer.data() == nullptr, "the add kernel accepts empty inputs and output without storage");
    }
}
