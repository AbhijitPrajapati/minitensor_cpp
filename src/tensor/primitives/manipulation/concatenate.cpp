#include "concatenate.hpp"

#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <minitensor/ops/manipulation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "tensor/core/axis.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/ops/apply_primitive.hpp"

namespace minitensor::detail
{
	ConcatenatePrimitive::ConcatenatePrimitive(Axis axis, Shape::size_type input_rank)
		: axis_(normalize_axis(axis, input_rank)), input_rank_(input_rank)
	{}

	Shape::size_type ConcatenatePrimitive::axis() const noexcept
	{
		return axis_;
	}

	std::string_view ConcatenatePrimitive::name() const noexcept
	{
		return "concatenate";
	}

	TensorSpec ConcatenatePrimitive::infer(std::span<const TensorSpec> inputs) const
	{
		if (inputs.empty())
		{
			throw std::invalid_argument{ "concatenate requires at least one input tensor" };
		}

		const TensorSpec& first = inputs.front();
		if (first.shape.rank() != input_rank_)
		{
			throw std::invalid_argument{
				"concatenate primitive was normalized for a different input rank" };
		}
		Extent concatenation_axis_extent = first.shape[axis_];

		for (const TensorSpec& spec : inputs.subspan(1))
		{
			if (spec.device != first.device)
			{
				throw std::invalid_argument{ "concatenate requires tensors on the same device" };
			}

			if (spec.dtype != first.dtype)
			{
				throw std::invalid_argument{ "concatenate requires matching dtypes" };
			}

			if (spec.shape.rank() != input_rank_)
			{
				throw std::invalid_argument{ "concatenate requires tensors with equal ranks" };
			}

			for (std::size_t axis = 0; axis < input_rank_; ++axis)
			{
				if (axis == axis_)
				{
					continue;
				}

				if (first.shape[axis] != spec.shape[axis])
				{
					throw std::invalid_argument{ "concatenate requires tensors to match in shape except on concatenation axis" };
				}
			}

			const Extent input_axis_extent = spec.shape[axis_];
			if (concatenation_axis_extent >
				std::numeric_limits<Extent>::max() - input_axis_extent)
			{
				throw std::overflow_error{ "concatenated extent overflow" };
			}
			concatenation_axis_extent += input_axis_extent;
		}

		std::vector<Extent> output_extents(
			first.shape.dimensions().begin(), first.shape.dimensions().end());
		output_extents[axis_] = concatenation_axis_extent;
		return TensorSpec
		{
			Shape{ std::move(output_extents) },
			first.dtype,
			first.device
		};
	}

	std::vector<std::optional<Tensor>> ConcatenatePrimitive::vjp(std::span<const Tensor> inputs, const Tensor&, const Tensor& output_cotangent) const
	{
		if (inputs.empty())
		{
			throw std::logic_error{ "concatenate VJP expects at least 1 input" };
		}

		std::vector<std::optional<Tensor>> contributions;
		contributions.reserve(inputs.size());

		Extent start = 0;
		for (const Tensor& input : inputs)
		{
			const Extent length = input.shape()[axis_];
			contributions.emplace_back(
				slice(output_cotangent, axis_, start, start + length, 1));
			start += length;
		}
		return contributions;
	}
}
