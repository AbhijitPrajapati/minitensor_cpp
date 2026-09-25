#include <minitensor/types.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <typeinfo>

#include "tensor/backend/cpu/cpu_buffer.hpp"
#include "tensor/backend/cpu/cpu_runtime.hpp"
#include "tensor/backend/cpu/detail/random_engine.hpp"
#include "tensor/backend/cpu/register_kernels.hpp"
#include "tensor/core/random.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/creation/normal.hpp"
#include "tensor/primitives/creation/uniform.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

#include "../support/test.hpp"

namespace minitensor::test
{
    namespace
    {
        detail::cpu::CpuBuffer &as_cpu_buffer(const detail::BufferRef &buffer)
        {
            expect(buffer != nullptr, "a CPU random kernel fixture requires a buffer");
            auto *cpu_buffer = dynamic_cast<detail::cpu::CpuBuffer *>(buffer.get());
            expect(cpu_buffer != nullptr, "a CPU random kernel fixture requires CPU storage");
            return *cpu_buffer;
        }
    }

    void run_cpu_random_test()
    {
        using detail::BufferRef;
        using detail::KernelFn;
        using detail::KernelKey;
        using detail::KernelRegistry;
        using detail::Layout;
        using detail::Materialization;
        using detail::MutableTensorView;
        using detail::NormalParameters;
        using detail::NormalPrimitive;
        using detail::RandomKey;
        using detail::TensorSpec;
        using detail::TensorView;
        using detail::UniformParameters;
        using detail::UniformPrimitive;
        using detail::cpu::CpuRuntime;
        using detail::cpu::open_unit_uniform;
        using detail::cpu::random_block;
        using detail::cpu::unit_uniform;

        constexpr std::array<std::uint32_t, 4> expected_zero_block{
            0x6627E8D5U,
            0xE169C58DU,
            0xBC57AC4CU,
            0x9B00DBD8U};
        expect(random_block(RandomKey{0, 0}, 0).words == expected_zero_block,
               "Philox matches its zero-key zero-counter reference vector");
        expect(random_block(RandomKey{0, 0}, 1).words != expected_zero_block &&
                   random_block(RandomKey{0, 1}, 0).words != expected_zero_block,
               "Philox separates block and stream counters");

        expect(unit_uniform(0) == 0.0F &&
                   unit_uniform(std::numeric_limits<std::uint32_t>::max()) < 1.0F,
               "Float32 unit conversion stays in the half-open unit interval");
        expect(open_unit_uniform(0) > 0.0 &&
                   open_unit_uniform(std::numeric_limits<std::uint32_t>::max()) < 1.0,
               "open unit conversion excludes both endpoints");

        KernelRegistry registry;
        detail::cpu::register_kernels(registry);
        const KernelKey uniform_key{typeid(UniformPrimitive), DeviceType::Cpu};
        const KernelKey normal_key{typeid(NormalPrimitive), DeviceType::Cpu};
        expect(registry.contains(uniform_key) && registry.contains(normal_key),
               "CPU registration installs both random creation kernels");

        CpuRuntime runtime;
        const KernelFn &uniform_kernel = registry.get(uniform_key);
        const KernelFn &normal_kernel = registry.get(normal_key);

        const TensorSpec uniform_spec{Shape{5}, DType::Float32, Device::cpu()};
        const BufferRef uniform_buffer = runtime.allocate(7 * sizeof(float));
        auto &uniform_cpu_buffer = as_cpu_buffer(uniform_buffer);
        auto *uniform_data = reinterpret_cast<float *>(uniform_cpu_buffer.data());
        std::fill_n(uniform_data, 7, -100.0F);

        const Materialization uniform_materialization{uniform_buffer, Layout{{1}, 1}};
        const MutableTensorView uniform_output{uniform_spec, uniform_materialization};
        const UniformPrimitive uniform_primitive{
            uniform_spec,
            UniformParameters{-2.0F, 3.0F},
            RandomKey{101, 7}};
        uniform_kernel(
            runtime,
            uniform_primitive,
            std::span<const TensorView>{},
            uniform_output);

        expect(uniform_data[0] == -100.0F && uniform_data[6] == -100.0F,
               "the uniform kernel respects the output layout offset and extent");
        expect(std::all_of(
                   uniform_data + 1,
                   uniform_data + 6,
                   [](float value)
                   {
                       return value >= -2.0F && value < 3.0F;
                   }),
               "the uniform kernel produces values inside its half-open interval");

        std::array<float, 5> first_uniform{};
        std::copy_n(uniform_data + 1, first_uniform.size(), first_uniform.begin());
        uniform_kernel(
            runtime,
            uniform_primitive,
            std::span<const TensorView>{},
            uniform_output);
        expect(std::equal(first_uniform.begin(), first_uniform.end(), uniform_data + 1),
               "the uniform kernel is deterministic for a fixed key");

        const TensorSpec normal_spec{Shape{4096}, DType::Float32, Device::cpu()};
        const BufferRef normal_buffer = runtime.allocate(normal_spec.shape.numel() * sizeof(float));
        auto &normal_cpu_buffer = as_cpu_buffer(normal_buffer);
        const Materialization normal_materialization{
            normal_buffer,
            Layout::contiguous(normal_spec.shape)};
        const MutableTensorView normal_output{normal_spec, normal_materialization};
        const NormalPrimitive normal_primitive{
            normal_spec,
            NormalParameters{1.5F, 0.75F},
            RandomKey{103, 11}};
        normal_kernel(
            runtime,
            normal_primitive,
            std::span<const TensorView>{},
            normal_output);

        const auto *normal_data =
            reinterpret_cast<const float *>(normal_cpu_buffer.data());
        double sum = 0.0;
        double squared_sum = 0.0;
        bool all_finite = true;
        for (std::size_t index = 0; index < normal_spec.shape.numel(); ++index)
        {
            const double value = normal_data[index];
            all_finite = all_finite && std::isfinite(value);
            sum += value;
            squared_sum += value * value;
        }
        const double sample_mean = sum / normal_spec.shape.numel();
        const double sample_variance =
            squared_sum / normal_spec.shape.numel() - sample_mean * sample_mean;
        const double sample_std_dev = std::sqrt(sample_variance);
        expect(all_finite && std::fabs(sample_mean - 1.5) < 0.1 &&
                   std::fabs(sample_std_dev - 0.75) < 0.1,
               "the normal kernel has the requested location and scale");

        const TensorSpec empty_spec{Shape{0}, DType::Float32, Device::cpu()};
        const BufferRef empty_buffer = runtime.allocate(0);
        auto &empty_cpu_buffer = as_cpu_buffer(empty_buffer);
        const Materialization empty_materialization{
            empty_buffer,
            Layout::contiguous(empty_spec.shape)};
        const MutableTensorView empty_output{empty_spec, empty_materialization};
        const UniformPrimitive empty_primitive{
            empty_spec,
            UniformParameters{0.0F, 1.0F},
            RandomKey{107, 13}};
        uniform_kernel(
            runtime,
            empty_primitive,
            std::span<const TensorView>{},
            empty_output);
        expect(empty_cpu_buffer.data() == nullptr,
               "random kernels accept empty outputs without accessing storage");
    }
}
