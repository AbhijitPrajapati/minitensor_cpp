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
#include "tensor/primitives/creation/full.hpp"
#include "tensor/primitives/elementwise/add.hpp"
#include "tensor/primitives/elementwise/divide.hpp"
#include "tensor/primitives/elementwise/multiply.hpp"
#include "tensor/primitives/elementwise/negate.hpp"
#include "tensor/primitives/elementwise/subtract.hpp"
#include "tensor/primitives/manipulation/concatenate.hpp"
#include "tensor/primitives/manipulation/contiguous.hpp"
#include "tensor/primitives/manipulation/reshape.hpp"
#include "tensor/primitives/reduction/sum.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

#include "../support/test.hpp"

namespace minitensor::test
{
	namespace
	{
		detail::cpu::CpuBuffer& as_cpu_buffer(const detail::BufferRef& buffer)
		{
			expect(buffer != nullptr, "a cpu kernel fixture requires a buffer");
			auto* cpu_buffer = dynamic_cast<detail::cpu::CpuBuffer*>(buffer.get());
			expect(cpu_buffer != nullptr, "a cpu kernel fixture requires cpu storage");
			return *cpu_buffer;
		}
	}

	void run_cpu_kernels_test()
	{
		using detail::AddPrimitive;
		using detail::BufferRef;
		using detail::ConcatenatePrimitive;
		using detail::ContiguousPrimitive;
		using detail::DividePrimitive;
		using detail::FullPrimitive;
		using detail::KernelFn;
		using detail::KernelKey;
		using detail::KernelRegistry;
		using detail::Layout;
		using detail::Materialization;
		using detail::MultiplyPrimitive;
		using detail::MutableTensorView;
		using detail::NegatePrimitive;
		using detail::ReshapePrimitive;
		using detail::SubtractPrimitive;
		using detail::SumPrimitive;
		using detail::TensorSpec;
		using detail::TensorView;
		using detail::cpu::CpuRuntime;

		KernelRegistry registry;
		detail::cpu::register_kernels(registry);

		KernelKey full_key{ typeid(FullPrimitive), DeviceType::Cpu };
		KernelKey add_key{ typeid(AddPrimitive), DeviceType::Cpu };
		KernelKey negate_key{ typeid(NegatePrimitive), DeviceType::Cpu };
		KernelKey subtract_key{ typeid(SubtractPrimitive), DeviceType::Cpu };
		KernelKey multiply_key{ typeid(MultiplyPrimitive), DeviceType::Cpu };
		KernelKey divide_key{ typeid(DividePrimitive), DeviceType::Cpu };
		KernelKey reshape_key{ typeid(ReshapePrimitive), DeviceType::Cpu };
		KernelKey contiguous_key{ typeid(ContiguousPrimitive), DeviceType::Cpu };
		KernelKey concatenate_key{ typeid(ConcatenatePrimitive), DeviceType::Cpu };
		KernelKey sum_key{ typeid(SumPrimitive), DeviceType::Cpu };
		expect(registry.contains(full_key), "cpu kernel registration installs the full kernel");
		expect(registry.contains(add_key), "cpu kernel registration installs the add kernel");
		expect(registry.contains(negate_key), "cpu kernel registration installs the negate kernel");
		expect(registry.contains(subtract_key), "cpu kernel registration installs the subtract kernel");
		expect(registry.contains(multiply_key), "cpu kernel registration installs the multiply kernel");
		expect(registry.contains(divide_key), "cpu kernel registration installs the divide kernel");
		expect(registry.contains(reshape_key), "cpu kernel registration installs the reshape kernel");
		expect(registry.contains(contiguous_key), "cpu kernel registration installs the contiguous kernel");
		expect(registry.contains(concatenate_key), "cpu kernel registration installs the concatenate kernel");
		expect(registry.contains(sum_key), "cpu kernel registration installs the sum kernel");

		CpuRuntime runtime;
		const KernelFn& full_kernel = registry.get(full_key);
		const KernelFn& add_kernel = registry.get(add_key);
		const KernelFn& reshape_kernel = registry.get(reshape_key);
		const KernelFn& concatenate_kernel = registry.get(concatenate_key);

		const TensorSpec full_spec{ Shape{2, 3}, DType::Float32, Device::cpu() };
		const BufferRef full_buffer = runtime.allocate(7 * sizeof(float));
		auto& full_cpu_buffer = as_cpu_buffer(full_buffer);
		auto* full_data = reinterpret_cast<float*>(full_cpu_buffer.data());
		std::fill_n(full_data, 7, -10.0F);

		const Layout full_layout{ {3, 1}, 1 };
		const Materialization full_materialization{ full_buffer, full_layout };
		const MutableTensorView full_output{ full_spec, full_materialization };
		const FullPrimitive full_primitive{ full_spec, 2.5F };
		full_kernel(runtime, full_primitive, std::span<const TensorView>{}, full_output);

		expect(full_data[0] == -10.0F, "the full kernel respects the output layout offset");
		expect(std::all_of(full_data + 1, full_data + 7, [](float value)
			{
				return value == 2.5F;
			}),
			   "the full kernel fills every logical output element");

		const TensorSpec empty_spec{ Shape{2, 0}, DType::Float32, Device::cpu() };
		const BufferRef empty_full_buffer = runtime.allocate(0);
		auto& empty_full_cpu_buffer = as_cpu_buffer(empty_full_buffer);
		const Materialization empty_full_materialization{
			empty_full_buffer,
			Layout::contiguous(empty_spec.shape) };
		const MutableTensorView empty_full_output{ empty_spec, empty_full_materialization };
		const FullPrimitive empty_full_primitive{ empty_spec, 4.0F };
		full_kernel(runtime, empty_full_primitive, std::span<const TensorView>{}, empty_full_output);
		expect(empty_full_cpu_buffer.data() == nullptr, "the full kernel accepts an empty output without storage");

		const TensorSpec lhs_spec{ Shape{2, 1}, DType::Float32, Device::cpu() };
		const TensorSpec rhs_spec{ Shape{1, 3}, DType::Float32, Device::cpu() };
		const TensorSpec sum_spec{ Shape{2, 3}, DType::Float32, Device::cpu() };
		const BufferRef lhs_buffer = runtime.allocate(2 * sizeof(float));
		const BufferRef rhs_buffer = runtime.allocate(3 * sizeof(float));
		const BufferRef sum_buffer = runtime.allocate(6 * sizeof(float));
		auto& lhs_cpu_buffer = as_cpu_buffer(lhs_buffer);
		auto& rhs_cpu_buffer = as_cpu_buffer(rhs_buffer);
		auto& sum_cpu_buffer = as_cpu_buffer(sum_buffer);

		const std::array<float, 2> lhs_values{ 1.0F, 2.0F };
		const std::array<float, 3> rhs_values{ 10.0F, 20.0F, 30.0F };
		std::copy(lhs_values.begin(), lhs_values.end(), reinterpret_cast<float*>(lhs_cpu_buffer.data()));
		std::copy(rhs_values.begin(), rhs_values.end(), reinterpret_cast<float*>(rhs_cpu_buffer.data()));

		const Materialization lhs_materialization{ lhs_buffer, Layout::contiguous(lhs_spec.shape) };
		const Materialization rhs_materialization{ rhs_buffer, Layout::contiguous(rhs_spec.shape) };
		const Materialization sum_materialization{ sum_buffer, Layout::contiguous(sum_spec.shape) };
		const std::array<TensorView, 2> add_inputs{
			TensorView{lhs_spec, lhs_materialization},
			TensorView{rhs_spec, rhs_materialization} };
		const MutableTensorView sum_output{ sum_spec, sum_materialization };
		const AddPrimitive add_primitive;
		add_kernel(runtime, add_primitive, add_inputs, sum_output);

		const std::array<float, 6> expected_sum{ 11.0F, 21.0F, 31.0F, 12.0F, 22.0F, 32.0F };
		const auto* sum_data = reinterpret_cast<const float*>(sum_cpu_buffer.data());
		expect(std::equal(expected_sum.begin(), expected_sum.end(), sum_data),
			   "the add kernel performs trailing-dimension broadcasting");

		const BufferRef empty_add_buffer = runtime.allocate(0);
		auto& empty_add_cpu_buffer = as_cpu_buffer(empty_add_buffer);
		const Materialization empty_add_materialization{
			empty_add_buffer,
			Layout::contiguous(empty_spec.shape) };
		const TensorView empty_input{ empty_spec, empty_add_materialization };
		const std::array<TensorView, 2> empty_inputs{ empty_input, empty_input };
		const MutableTensorView empty_sum_output{ empty_spec, empty_add_materialization };
		add_kernel(runtime, add_primitive, empty_inputs, empty_sum_output);
		expect(empty_add_cpu_buffer.data() == nullptr, "the add kernel accepts empty inputs and output without storage");

		const TensorSpec strided_copy_input_spec{
			Shape{2, 3}, DType::Float32, Device::cpu() };
		const TensorSpec strided_copy_output_spec{
			Shape{3, 2}, DType::Float32, Device::cpu() };
		const BufferRef strided_copy_input_buffer = runtime.allocate(6 * sizeof(float));
		const BufferRef strided_copy_output_buffer = runtime.allocate(7 * sizeof(float));
		auto& strided_copy_input_cpu_buffer = as_cpu_buffer(strided_copy_input_buffer);
		auto& strided_copy_output_cpu_buffer = as_cpu_buffer(strided_copy_output_buffer);
		const std::array<float, 6> strided_copy_physical_values{
			1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F };
		std::copy(
			strided_copy_physical_values.begin(),
			strided_copy_physical_values.end(),
			reinterpret_cast<float*>(strided_copy_input_cpu_buffer.data()));
		auto* strided_copy_output_data =
			reinterpret_cast<float*>(strided_copy_output_cpu_buffer.data());
		std::fill_n(strided_copy_output_data, 7, -1.0F);

		const Materialization strided_copy_input_materialization{
			strided_copy_input_buffer, Layout{{1, 2}} };
		const Materialization strided_copy_output_materialization{
			strided_copy_output_buffer,
			Layout::contiguous(strided_copy_output_spec.shape, 1) };
		const TensorView strided_copy_input{
			strided_copy_input_spec, strided_copy_input_materialization };
		const std::array<TensorView, 1> strided_copy_inputs{ strided_copy_input };
		const MutableTensorView strided_copy_output{
			strided_copy_output_spec, strided_copy_output_materialization };
		const ReshapePrimitive reshape_primitive{ strided_copy_output_spec.shape };
		reshape_kernel(
			runtime,
			reshape_primitive,
			strided_copy_inputs,
			strided_copy_output);

		const std::array<float, 7> expected_strided_copy{
			-1.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
		expect(std::equal(
			expected_strided_copy.begin(),
			expected_strided_copy.end(),
			strided_copy_output_data),
			"the copy kernel preserves logical order across strided reshape inputs");

		const TensorSpec concatenate_column_spec{
			Shape{2, 1}, DType::Float32, Device::cpu() };
		const TensorSpec concatenate_empty_spec{
			Shape{2, 0}, DType::Float32, Device::cpu() };
		const TensorSpec concatenate_block_spec{
			Shape{2, 2}, DType::Float32, Device::cpu() };
		const TensorSpec concatenate_output_spec{
			Shape{2, 3}, DType::Float32, Device::cpu() };
		const BufferRef concatenate_column_buffer = runtime.allocate(2 * sizeof(float));
		const BufferRef concatenate_empty_buffer = runtime.allocate(0);
		const BufferRef concatenate_block_buffer = runtime.allocate(4 * sizeof(float));
		const BufferRef concatenate_output_buffer = runtime.allocate(7 * sizeof(float));
		auto& concatenate_column_cpu_buffer = as_cpu_buffer(concatenate_column_buffer);
		auto& concatenate_block_cpu_buffer = as_cpu_buffer(concatenate_block_buffer);
		auto& concatenate_output_cpu_buffer = as_cpu_buffer(concatenate_output_buffer);
		const std::array<float, 2> concatenate_column_values{ 1.0F, 4.0F };
		const std::array<float, 4> concatenate_block_physical_values{
			2.0F, 5.0F, 3.0F, 6.0F };
		std::copy(
			concatenate_column_values.begin(),
			concatenate_column_values.end(),
			reinterpret_cast<float*>(concatenate_column_cpu_buffer.data()));
		std::copy(
			concatenate_block_physical_values.begin(),
			concatenate_block_physical_values.end(),
			reinterpret_cast<float*>(concatenate_block_cpu_buffer.data()));
		auto* concatenate_output_data =
			reinterpret_cast<float*>(concatenate_output_cpu_buffer.data());
		std::fill_n(concatenate_output_data, 7, -1.0F);

		const Materialization concatenate_column_materialization{
			concatenate_column_buffer, Layout::contiguous(concatenate_column_spec.shape) };
		const Materialization concatenate_empty_materialization{
			concatenate_empty_buffer, Layout::contiguous(concatenate_empty_spec.shape) };
		const Materialization concatenate_block_materialization{
			concatenate_block_buffer, Layout{{1, 2}} };
		const Materialization concatenate_output_materialization{
			concatenate_output_buffer,
			Layout::contiguous(concatenate_output_spec.shape, 1) };
		const std::array<TensorView, 3> concatenate_inputs{
			TensorView{concatenate_column_spec, concatenate_column_materialization},
			TensorView{concatenate_empty_spec, concatenate_empty_materialization},
			TensorView{concatenate_block_spec, concatenate_block_materialization} };
		const MutableTensorView concatenate_output{
			concatenate_output_spec, concatenate_output_materialization };
		const ConcatenatePrimitive concatenate_primitive{ 1, 2 };
		concatenate_kernel(
			runtime,
			concatenate_primitive,
			concatenate_inputs,
			concatenate_output);

		const std::array<float, 7> expected_concatenate{
			-1.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
		expect(std::equal(
			expected_concatenate.begin(),
			expected_concatenate.end(),
			concatenate_output_data),
			"the concatenate kernel copies strided and empty inputs into output sections");
	}
}
