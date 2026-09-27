#include <minitensor/types.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <typeinfo>

#include <gtest/gtest.h>

#include "tensor/backend/cpu/cpu_buffer.hpp"
#include "tensor/backend/cpu/cpu_runtime.hpp"
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

namespace minitensor::test
{
	namespace
	{
		float* float_data(const detail::BufferRef& buffer)
		{
			auto* cpu_buffer = dynamic_cast<detail::cpu::CpuBuffer*>(buffer.get());
			if (cpu_buffer == nullptr)
			{
				throw std::logic_error{ "CPU random test requires CPU storage" };
			}
			return reinterpret_cast<float*>(cpu_buffer->data());
		}
	}

	TEST(CpuRandomKernelTest, UniformIsDeterministicForAKeyAndHonorsOutputLayout)
	{
		detail::KernelRegistry registry;
		detail::cpu::register_kernels(registry);
		detail::cpu::CpuRuntime runtime;
		const detail::KernelFn& kernel = registry.get({
			typeid(detail::UniformPrimitive), DeviceType::Cpu });
		const detail::TensorSpec spec{ Shape{ 5 }, DType::Float32, Device::cpu() };
		const detail::BufferRef buffer = runtime.allocate(7 * sizeof(float));
		std::fill_n(float_data(buffer), 7, -100.0F);
		const detail::Materialization storage{ buffer, detail::Layout{ { 1 }, 1 } };
		const detail::MutableTensorView output{ spec, storage };
		const detail::UniformPrimitive primitive{
			spec,
			detail::UniformParameters{ -2.0F, 3.0F },
			detail::RandomKey{ 101, 7 } };

		kernel(runtime, primitive, std::span<const detail::TensorView>{}, output);
		EXPECT_FLOAT_EQ(float_data(buffer)[0], -100.0F);
		EXPECT_FLOAT_EQ(float_data(buffer)[6], -100.0F);
		EXPECT_TRUE(std::all_of(
			float_data(buffer) + 1,
			float_data(buffer) + 6,
			[](float value) { return value >= -2.0F && value < 3.0F; }));

		std::array<float, 5> first{};
		std::copy_n(float_data(buffer) + 1, first.size(), first.begin());
		kernel(runtime, primitive, std::span<const detail::TensorView>{}, output);
		EXPECT_TRUE(std::equal(first.begin(), first.end(), float_data(buffer) + 1));
	}

	TEST(CpuRandomKernelTest, NormalHasReasonableFiniteMoments)
	{
		detail::KernelRegistry registry;
		detail::cpu::register_kernels(registry);
		detail::cpu::CpuRuntime runtime;
		const detail::KernelFn& kernel = registry.get({
			typeid(detail::NormalPrimitive), DeviceType::Cpu });
		const detail::TensorSpec spec{ Shape{ 4096 }, DType::Float32, Device::cpu() };
		const detail::BufferRef buffer = runtime.allocate(spec.shape.numel() * sizeof(float));
		const detail::Materialization storage{
			buffer, detail::Layout::contiguous(spec.shape) };
		const detail::MutableTensorView output{ spec, storage };
		const detail::NormalPrimitive primitive{
			spec,
			detail::NormalParameters{ 1.5F, 0.75F },
			detail::RandomKey{ 103, 11 } };
		kernel(runtime, primitive, std::span<const detail::TensorView>{}, output);

		double sum = 0.0;
		double squared_sum = 0.0;
		for (std::size_t index = 0; index < spec.shape.numel(); ++index)
		{
			const double value = float_data(buffer)[index];
			ASSERT_TRUE(std::isfinite(value));
			sum += value;
			squared_sum += value * value;
		}
		const double mean = sum / spec.shape.numel();
		const double variance = squared_sum / spec.shape.numel() - mean * mean;
		EXPECT_NEAR(mean, 1.5, 0.1);
		EXPECT_NEAR(std::sqrt(variance), 0.75, 0.1);
	}

	TEST(CpuRandomKernelTest, AcceptsEmptyOutputs)
	{
		detail::KernelRegistry registry;
		detail::cpu::register_kernels(registry);
		detail::cpu::CpuRuntime runtime;
		const detail::KernelFn& kernel = registry.get({
			typeid(detail::UniformPrimitive), DeviceType::Cpu });
		const detail::TensorSpec spec{ Shape{ 0 }, DType::Float32, Device::cpu() };
		const detail::BufferRef buffer = runtime.allocate(0);
		const detail::Materialization storage{
			buffer, detail::Layout::contiguous(spec.shape) };
		const detail::MutableTensorView output{ spec, storage };
		const detail::UniformPrimitive primitive{
			spec,
			detail::UniformParameters{ 0.0F, 1.0F },
			detail::RandomKey{ 107, 13 } };
		EXPECT_NO_THROW(kernel(
			runtime, primitive, std::span<const detail::TensorView>{}, output));
	}
}
