#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <span>
#include <utility>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/detail/buffer_access.hpp"
#include "tensor/backend/cpu/iteration/elementwise.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
	template <CpuElement T>
	void copy_strided(
		const Shape& shape,
		const T* source_data,
		const Layout& source_layout,
		T* destination_data,
		const Layout& destination_layout)
	{
		assert(source_layout.rank() == shape.rank());
		assert(destination_layout.rank() == shape.rank());

		if (shape.numel() == 0)
		{
			return;
		}

		// Contiguous fast path
		if (source_layout.is_contiguous(shape) &&
			destination_layout.is_contiguous(shape))
		{
			std::copy_n(
				source_data + static_cast<std::size_t>(source_layout.offset()),
				shape.numel(),
				destination_data +
					static_cast<std::size_t>(destination_layout.offset()));
			return;
		}

		const std::array<Layout, 2> layouts{
			source_layout,
			destination_layout };
		const ElementwisePlan iteration(shape, layouts);
		iteration.for_each_run(
			[&](Shape::size_type,
			std::span<const Layout::offset_type> offsets,
			std::span<const Layout::stride_type> strides,
			Shape::size_type run_size)
			{
				Layout::offset_type source_offset = offsets[0];
				Layout::offset_type destination_offset = offsets[1];
				const Layout::stride_type source_stride = strides[0];
				const Layout::stride_type destination_stride = strides[1];

				// Fast path for single strides
				if (source_stride == 1 && destination_stride == 1)
				{
					assert(source_offset >= 0);
					assert(destination_offset >= 0);
					std::copy_n(
						source_data + static_cast<std::size_t>(source_offset),
						run_size,
						destination_data +
							static_cast<std::size_t>(destination_offset));
					return;
				}

				for (Shape::size_type i = 0; i < run_size; ++i)
				{
					assert(source_offset >= 0);
					assert(destination_offset >= 0);
					destination_data[static_cast<std::size_t>(destination_offset)] =
						source_data[static_cast<std::size_t>(source_offset)];
					source_offset += source_stride;
					destination_offset += destination_stride;
				}
			});
	}

	template <CpuElement T>
	void copy_to_contiguous(const TensorView& input, MutableTensorView output)
	{
		assert(input.dtype() == ElementDType<T>::value);
		assert(output.dtype() == ElementDType<T>::value);
		assert(input.shape().numel() == output.shape().numel());
		assert(output.layout().is_contiguous(output.shape()));

		if (output.shape().numel() == 0)
		{
			return;
		}

		const Layout destination_layout = Layout::contiguous(input.shape(), output.layout().offset());
		copy_strided(
			input.shape(),
			data<T>(input),
			input.layout(),
			data<T>(output),
			destination_layout);
	}

	template <CpuElement T>
	void concatenate_copy(
		std::span<const TensorView> inputs,
		MutableTensorView output,
		Shape::size_type axis)
	{
		assert(!inputs.empty());
		assert(output.dtype() == ElementDType<T>::value);
		assert(output.layout().is_contiguous(output.shape()));
		assert(axis < output.shape().rank());

		if (output.shape().numel() == 0)
		{
			return;
		}

		T* output_data = data<T>(output);
		Extent axis_start = 0;

		for (const TensorView& input : inputs)
		{
			assert(input.dtype() == ElementDType<T>::value);
			assert(input.shape().rank() == output.shape().rank());

			const Extent input_axis_extent = input.shape()[axis];
			if (input.shape().numel() != 0)
			{
				// Find layout for corresponding output section
				std::vector<Layout::stride_type> output_section_strides(
					output.layout().strides().begin(),
					output.layout().strides().end());
				const Layout::offset_type output_section_offset =
					output.layout().offset() +
					axis_start * output.layout().stride(axis);
				const Layout output_section_layout{
					std::move(output_section_strides), output_section_offset };

				copy_strided(
					input.shape(),
					data<T>(input),
					input.layout(),
					output_data,
					output_section_layout);
			}
			axis_start += input_axis_extent;
		}

		assert(axis_start == output.shape()[axis]);
	}
}
