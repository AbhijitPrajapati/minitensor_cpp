#include <minitensor/types.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <typeinfo>

#include <gtest/gtest.h>

#include "tensor/backend/cpu/cpu_buffer.hpp"
#include "tensor/backend/cpu/cpu_runtime.hpp"
#include "tensor/backend/cpu/register_kernels.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/primitives/creation/full.hpp"
#include "tensor/primitives/elementwise/add.hpp"
#include "tensor/primitives/manipulation/concatenate.hpp"
#include "tensor/primitives/manipulation/reshape.hpp"
#include "tensor/primitives/manipulation/slice.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::test
{
	namespace
	{
		detail::cpu::CpuBuffer& as_cpu_buffer(const detail::BufferRef& buffer)
		{
			auto* cpu_buffer = dynamic_cast<detail::cpu::CpuBuffer*>(buffer.get());
			if (cpu_buffer == nullptr)
			{
				throw std::logic_error{ "CPU kernel test requires CPU storage" };
			}
			return *cpu_buffer;
		}

		float* float_data(const detail::BufferRef& buffer)
		{
			return reinterpret_cast<float*>(as_cpu_buffer(buffer).data());
		}
	}

	TEST(CpuCreationKernelTest, WritesOffsetAndEmptyOutputs)
	{
		detail::KernelRegistry registry;
		detail::cpu::register_kernels(registry);
		detail::cpu::CpuRuntime runtime;
		const detail::KernelFn& kernel = registry.get({
			typeid(detail::FullPrimitive), DeviceType::Cpu });

		const detail::TensorSpec spec{ Shape{ 2, 3 }, DType::Float32, Device::cpu() };
		const detail::BufferRef buffer = runtime.allocate(7 * sizeof(float));
		std::fill_n(float_data(buffer), 7, -10.0F);
		const detail::Materialization storage{ buffer, detail::Layout{ { 3, 1 }, 1 } };
		const detail::MutableTensorView output{ spec, storage };
		const detail::FullPrimitive primitive{ spec, 2.5F };
		kernel(runtime, primitive, std::span<const detail::TensorView>{}, output);

		const std::array<float, 7> expected{
			-10.0F, 2.5F, 2.5F, 2.5F, 2.5F, 2.5F, 2.5F };
		EXPECT_TRUE(std::equal(expected.begin(), expected.end(), float_data(buffer)));

		const detail::TensorSpec empty_spec{ Shape{ 2, 0 }, DType::Float32, Device::cpu() };
		const detail::BufferRef empty_buffer = runtime.allocate(0);
		const detail::Materialization empty_storage{
			empty_buffer, detail::Layout::contiguous(empty_spec.shape) };
		const detail::MutableTensorView empty_output{ empty_spec, empty_storage };
		const detail::FullPrimitive empty_primitive{ empty_spec, 4.0F };
		EXPECT_NO_THROW(kernel(
			runtime, empty_primitive, std::span<const detail::TensorView>{}, empty_output));
	}

	TEST(CpuElementwiseKernelTest, HandlesBroadcastNegativeStridesAndOutputOffsets)
	{
		detail::KernelRegistry registry;
		detail::cpu::register_kernels(registry);
		detail::cpu::CpuRuntime runtime;
		const detail::KernelFn& kernel = registry.get({
			typeid(detail::AddPrimitive), DeviceType::Cpu });

		const detail::TensorSpec lhs_spec{ Shape{ 2, 3 }, DType::Float32, Device::cpu() };
		const detail::TensorSpec rhs_spec{ Shape{ 1, 3 }, DType::Float32, Device::cpu() };
		const detail::TensorSpec output_spec{ Shape{ 2, 3 }, DType::Float32, Device::cpu() };
		const detail::BufferRef lhs_buffer = runtime.allocate(6 * sizeof(float));
		const detail::BufferRef rhs_buffer = runtime.allocate(3 * sizeof(float));
		const detail::BufferRef output_buffer = runtime.allocate(7 * sizeof(float));
		const std::array<float, 6> lhs_physical{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
		const std::array<float, 3> rhs_values{ 10.0F, 20.0F, 30.0F };
		std::copy(lhs_physical.begin(), lhs_physical.end(), float_data(lhs_buffer));
		std::copy(rhs_values.begin(), rhs_values.end(), float_data(rhs_buffer));
		std::fill_n(float_data(output_buffer), 7, -1.0F);

		const detail::Materialization lhs_storage{
			lhs_buffer, detail::Layout{ { -3, 1 }, 3 } };
		const detail::Materialization rhs_storage{
			rhs_buffer, detail::Layout::contiguous(rhs_spec.shape) };
		const detail::Materialization output_storage{
			output_buffer, detail::Layout::contiguous(output_spec.shape, 1) };
		const std::array<detail::TensorView, 2> inputs{
			detail::TensorView{ lhs_spec, lhs_storage },
			detail::TensorView{ rhs_spec, rhs_storage }
		};
		const detail::MutableTensorView output{ output_spec, output_storage };
		kernel(runtime, detail::AddPrimitive{}, inputs, output);

		const std::array<float, 7> expected{
			-1.0F, 14.0F, 25.0F, 36.0F, 11.0F, 22.0F, 33.0F };
		EXPECT_TRUE(std::equal(expected.begin(), expected.end(), float_data(output_buffer)));
	}

	TEST(CpuCopyKernelTest, CopiesNegativeStridedInputsInLogicalOrder)
	{
		detail::KernelRegistry registry;
		detail::cpu::register_kernels(registry);
		detail::cpu::CpuRuntime runtime;
		const detail::KernelFn& kernel = registry.get({
			typeid(detail::ReshapePrimitive), DeviceType::Cpu });

		const detail::TensorSpec input_spec{ Shape{ 2, 3 }, DType::Float32, Device::cpu() };
		const detail::TensorSpec output_spec{ Shape{ 3, 2 }, DType::Float32, Device::cpu() };
		const detail::BufferRef input_buffer = runtime.allocate(6 * sizeof(float));
		const detail::BufferRef output_buffer = runtime.allocate(7 * sizeof(float));
		const std::array<float, 6> physical{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
		std::copy(physical.begin(), physical.end(), float_data(input_buffer));
		std::fill_n(float_data(output_buffer), 7, -1.0F);

		const detail::Materialization input_storage{
			input_buffer, detail::Layout{ { -3, 1 }, 3 } };
		const detail::Materialization output_storage{
			output_buffer, detail::Layout::contiguous(output_spec.shape, 1) };
		const std::array<detail::TensorView, 1> inputs{
			detail::TensorView{ input_spec, input_storage } };
		const detail::MutableTensorView output{ output_spec, output_storage };
		kernel(
			runtime,
			detail::ReshapePrimitive{ output_spec.shape },
			inputs,
			output);

		const std::array<float, 7> expected{
			-1.0F, 4.0F, 5.0F, 6.0F, 1.0F, 2.0F, 3.0F };
		EXPECT_TRUE(std::equal(expected.begin(), expected.end(), float_data(output_buffer)));
	}

	TEST(CpuConcatenateKernelTest, CopiesStridedAndEmptySections)
	{
		detail::KernelRegistry registry;
		detail::cpu::register_kernels(registry);
		detail::cpu::CpuRuntime runtime;
		const detail::KernelFn& kernel = registry.get({
			typeid(detail::ConcatenatePrimitive), DeviceType::Cpu });

		const detail::TensorSpec column_spec{ Shape{ 2, 1 }, DType::Float32, Device::cpu() };
		const detail::TensorSpec empty_spec{ Shape{ 2, 0 }, DType::Float32, Device::cpu() };
		const detail::TensorSpec block_spec{ Shape{ 2, 2 }, DType::Float32, Device::cpu() };
		const detail::TensorSpec output_spec{ Shape{ 2, 3 }, DType::Float32, Device::cpu() };
		const detail::BufferRef column_buffer = runtime.allocate(2 * sizeof(float));
		const detail::BufferRef empty_buffer = runtime.allocate(0);
		const detail::BufferRef block_buffer = runtime.allocate(4 * sizeof(float));
		const detail::BufferRef output_buffer = runtime.allocate(7 * sizeof(float));
		const std::array<float, 2> column_values{ 1.0F, 4.0F };
		const std::array<float, 4> block_physical{ 2.0F, 5.0F, 3.0F, 6.0F };
		std::copy(column_values.begin(), column_values.end(), float_data(column_buffer));
		std::copy(block_physical.begin(), block_physical.end(), float_data(block_buffer));
		std::fill_n(float_data(output_buffer), 7, -1.0F);

		const detail::Materialization column_storage{
			column_buffer, detail::Layout::contiguous(column_spec.shape) };
		const detail::Materialization empty_storage{
			empty_buffer, detail::Layout::contiguous(empty_spec.shape) };
		const detail::Materialization block_storage{
			block_buffer, detail::Layout{ { 1, 2 } } };
		const detail::Materialization output_storage{
			output_buffer, detail::Layout::contiguous(output_spec.shape, 1) };
		const std::array<detail::TensorView, 3> inputs{
			detail::TensorView{ column_spec, column_storage },
			detail::TensorView{ empty_spec, empty_storage },
			detail::TensorView{ block_spec, block_storage }
		};
		const detail::MutableTensorView output{ output_spec, output_storage };
		kernel(runtime, detail::ConcatenatePrimitive{ 1, 2 }, inputs, output);

		const std::array<float, 7> expected{
			-1.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
		EXPECT_TRUE(std::equal(expected.begin(), expected.end(), float_data(output_buffer)));
	}

	TEST(CpuSliceScatterKernelTest, ZeroFillsGapsForNegativeSteps)
	{
		detail::KernelRegistry registry;
		detail::cpu::register_kernels(registry);
		detail::cpu::CpuRuntime runtime;
		const detail::KernelFn& kernel = registry.get({
			typeid(detail::SliceScatterPrimitive), DeviceType::Cpu });

		const detail::TensorSpec input_spec{ Shape{ 2, 3 }, DType::Float32, Device::cpu() };
		const detail::TensorSpec output_spec{ Shape{ 2, 5 }, DType::Float32, Device::cpu() };
		const detail::BufferRef input_buffer = runtime.allocate(6 * sizeof(float));
		const detail::BufferRef output_buffer = runtime.allocate(11 * sizeof(float));
		const std::array<float, 6> input_values{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
		std::copy(input_values.begin(), input_values.end(), float_data(input_buffer));
		std::fill_n(float_data(output_buffer), 11, -1.0F);

		const detail::Materialization input_storage{
			input_buffer, detail::Layout::contiguous(input_spec.shape) };
		const detail::Materialization output_storage{
			output_buffer, detail::Layout::contiguous(output_spec.shape, 1) };
		const std::array<detail::TensorView, 1> inputs{
			detail::TensorView{ input_spec, input_storage } };
		const detail::MutableTensorView output{ output_spec, output_storage };
		const detail::SliceParameters parameters{
			1, std::nullopt, std::nullopt, -2, output_spec.shape };
		kernel(
			runtime,
			detail::SliceScatterPrimitive{ output_spec.shape, parameters },
			inputs,
			output);

		const std::array<float, 11> expected{
			-1.0F,
			3.0F, 0.0F, 2.0F, 0.0F, 1.0F,
			6.0F, 0.0F, 5.0F, 0.0F, 4.0F
		};
		EXPECT_TRUE(std::equal(expected.begin(), expected.end(), float_data(output_buffer)));
	}
}
