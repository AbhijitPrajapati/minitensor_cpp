#pragma once

#include <cassert>
#include <concepts>
#include <functional>
#include <span>
#include <type_traits>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/detail/buffer_access.hpp"
#include "tensor/backend/cpu/iteration/reduction.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
	template <CpuElement T, typename Accumulator, typename Operation>
		requires std::invocable<Operation&, Accumulator, T>&&
	std::convertible_to<
		std::invoke_result_t<Operation&, Accumulator, T>,
		Accumulator>&&
		std::convertible_to<Accumulator, T>
		void reduce_values(
			const TensorView& input,
			MutableTensorView output,
			std::span<const Shape::size_type> axes,
			Accumulator initial_value,
			Operation&& operation)
	{
		assert(input.dtype() == ElementDType<T>::value);
		assert(output.dtype() == ElementDType<T>::value);
		assert(output.layout().is_contiguous(output.shape()));

		const ReductionPlan plan{ input.shape(), input.layout(), axes };
		assert(plan.output_size() == output.shape().numel());
		if (plan.output_size() == 0)
		{
			return;
		}

		const T* input_data = data<T>(input);
		T* output_data = data<T>(output);
		const auto output_offset = static_cast<std::size_t>(output.layout().offset());

		plan.for_each_output(
			[&](Shape::size_type output_linear,
			const ReductionPlan::ReductionRange& range)
			{
				Accumulator accumulator = initial_value;
				range.for_each_run(
					[&](const ReductionPlan::Run& run)
					{
						Layout::offset_type input_offset = run.offset;
						for (Shape::size_type i = 0; i < run.size; ++i)
						{
							assert(input_offset >= 0);
							accumulator = static_cast<Accumulator>(std::invoke(
								operation,
								accumulator,
								input_data[static_cast<std::size_t>(input_offset)]));
							input_offset += run.stride;
						}
					});
				output_data[output_offset + output_linear] =
					static_cast<T>(accumulator);
			});
	}
}
