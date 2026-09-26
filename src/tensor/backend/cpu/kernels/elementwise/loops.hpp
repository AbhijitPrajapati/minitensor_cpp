#pragma once

#include <array>
#include <cassert>
#include <concepts>
#include <functional>
#include <span>
#include <type_traits>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/detail/buffer_access.hpp"
#include "tensor/backend/cpu/iteration/elementwise.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
	template <CpuElement T, typename Operation>
		requires std::invocable<Operation&, T>&& std::convertible_to<std::invoke_result_t<Operation&, T>, T>
	void unary_elementwise(const TensorView& input, MutableTensorView output, Operation&& operation)
	{
		assert(input.dtype() == ElementDType<T>::value);
		assert(output.dtype() == ElementDType<T>::value);

		const Shape& output_shape = output.shape();
		assert(input.shape() == output_shape);
		assert(output.layout().is_contiguous(output_shape));

		if (output_shape.numel() == 0)
		{
			return;
		}

		const T* input_data = data<T>(input);
		T* output_data = data<T>(output);
		const auto input_offset = static_cast<std::size_t>(input.layout().offset());
		const auto output_offset = static_cast<std::size_t>(output.layout().offset());
		const Shape::size_type numel = output_shape.numel();

		// Contiguous fast path
		if (input.layout().is_contiguous(output_shape))
		{
			for (Shape::size_type i = 0; i < numel; ++i)
			{
				output_data[output_offset + i] =
					static_cast<T>(std::invoke(operation, input_data[input_offset + i]));
			}
			return;
		}

		// Regular strided path
		const std::array<Layout, 1> layouts{ input.layout() };
		const ElementwisePlan iteration(output_shape, layouts);
		iteration.for_each_run(
			[&](Shape::size_type linear,
			std::span<const Layout::offset_type> offsets,
			std::span<const Layout::stride_type> strides,
			Shape::size_type run_size)
			{
				Layout::offset_type input_offset = offsets[0];
				const Layout::stride_type input_stride = strides[0];
				T* run_output = output_data + output_offset + linear;

				for (Shape::size_type i = 0; i < run_size; ++i)
				{
					assert(input_offset >= 0);
					run_output[i] = static_cast<T>(std::invoke(
						operation,
						input_data[static_cast<std::size_t>(input_offset)]));
					input_offset += input_stride;
				}
			});
	}

	template <CpuElement T, typename Operation>
		requires std::invocable<Operation&, T, T>&& std::convertible_to<std::invoke_result_t<Operation&, T, T>, T>
	void binary_elementwise(const TensorView& lhs, const TensorView& rhs, MutableTensorView output, Operation&& operation)
	{
		assert(lhs.dtype() == ElementDType<T>::value);
		assert(rhs.dtype() == ElementDType<T>::value);
		assert(output.dtype() == ElementDType<T>::value);

		const Shape& output_shape = output.shape();
		assert(output.layout().is_contiguous(output_shape));

		if (output_shape.numel() == 0)
		{
			return;
		}

		const T* lhs_data = data<T>(lhs);
		const T* rhs_data = data<T>(rhs);
		T* output_data = data<T>(output);
		const auto output_offset = static_cast<std::size_t>(output.layout().offset());
		const Shape::size_type numel = output_shape.numel();

		// No-broadcast both-contiguous fast path
		if (lhs.shape() == output_shape && rhs.shape() == output_shape &&
			lhs.layout().is_contiguous(output_shape) &&
			rhs.layout().is_contiguous(output_shape))
		{
			const auto lhs_offset = static_cast<std::size_t>(lhs.layout().offset());
			const auto rhs_offset = static_cast<std::size_t>(rhs.layout().offset());
			for (Shape::size_type i = 0; i < numel; ++i)
			{
				output_data[output_offset + i] = static_cast<T>(std::invoke(
					operation,
					lhs_data[lhs_offset + i],
					rhs_data[rhs_offset + i]));
			}
			return;
		}

		// Lhs scalar rhs contiguous fast path
		if (lhs.shape().numel() == 1 && rhs.shape() == output_shape &&
			rhs.layout().is_contiguous(output_shape))
		{
			const T lhs_value =
				lhs_data[static_cast<std::size_t>(lhs.layout().offset())];
			const auto rhs_offset = static_cast<std::size_t>(rhs.layout().offset());
			for (Shape::size_type i = 0; i < numel; ++i)
			{
				output_data[output_offset + i] = static_cast<T>(std::invoke(
					operation,
					lhs_value,
					rhs_data[rhs_offset + i]));
			}
			return;
		}

		// rhs scalar lhs contiguous fast path
		if (rhs.shape().numel() == 1 && lhs.shape() == output_shape &&
			lhs.layout().is_contiguous(output_shape))
		{
			const auto lhs_offset = static_cast<std::size_t>(lhs.layout().offset());
			const T rhs_value =
				rhs_data[static_cast<std::size_t>(rhs.layout().offset())];
			for (Shape::size_type i = 0; i < numel; ++i)
			{
				output_data[output_offset + i] = static_cast<T>(std::invoke(
					operation,
					lhs_data[lhs_offset + i],
					rhs_value));
			}
			return;
		}

		// Regular strided path
		const std::array<Layout, 2> layouts{
			lhs.layout().broadcasted_to(lhs.shape(), output_shape),
			rhs.layout().broadcasted_to(rhs.shape(), output_shape) };

		const Layout& lhs_layout = layouts[0];
		const Layout& rhs_layout = layouts[1];
		const auto lhs_offset = static_cast<std::size_t>(lhs_layout.offset());
		const auto rhs_offset = static_cast<std::size_t>(rhs_layout.offset());
		const ElementwisePlan iteration(output_shape, layouts);
		iteration.for_each_run(
			[&](Shape::size_type linear,
			std::span<const Layout::offset_type> offsets,
			std::span<const Layout::stride_type> strides,
			Shape::size_type run_size)
			{
				Layout::offset_type lhs_offset = offsets[0];
				Layout::offset_type rhs_offset = offsets[1];
				const Layout::stride_type lhs_stride = strides[0];
				const Layout::stride_type rhs_stride = strides[1];
				T* run_output = output_data + output_offset + linear;
				for (Shape::size_type i = 0; i < run_size; ++i)
				{
					assert(lhs_offset >= 0);
					assert(rhs_offset >= 0);
					run_output[i] = static_cast<T>(std::invoke(
						operation,
						lhs_data[static_cast<std::size_t>(lhs_offset)],
						rhs_data[static_cast<std::size_t>(rhs_offset)]));
					lhs_offset += lhs_stride;
					rhs_offset += rhs_stride;
				}
			});
	}
}
