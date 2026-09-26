#include <minitensor/ops/manipulation.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "apply_primitive.hpp"
#include "tensor/core/axis.hpp"
#include "tensor/primitives/manipulation/broadcast_to.hpp"
#include "tensor/primitives/manipulation/concatenate.hpp"
#include "tensor/primitives/manipulation/contiguous.hpp"
#include "tensor/primitives/manipulation/permute.hpp"
#include "tensor/primitives/manipulation/reshape.hpp"
#include "tensor/primitives/manipulation/slice.hpp"

namespace minitensor
{
	namespace
	{
		[[nodiscard]] Extent flattened_extent(
			const Shape& shape,
			Shape::size_type start_axis,
			Shape::size_type end_axis)
		{
			for (Shape::size_type axis = start_axis; axis <= end_axis; ++axis)
			{
				if (shape[axis] == 0)
				{
					return 0;
				}
			}

			Extent extent = 1;
			constexpr Extent max_extent = std::numeric_limits<Extent>::max();
			for (Shape::size_type axis = start_axis; axis <= end_axis; ++axis)
			{
				if (extent > max_extent / shape[axis])
				{
					throw std::overflow_error{ "flattened extent overflow" };
				}
				extent *= shape[axis];
			}
			return extent;
		}
	}

	Tensor contiguous(const Tensor& input)
	{
		return detail::apply_primitive(
			std::make_unique<detail::ContiguousPrimitive>(), input);
	}

	Tensor permute(const Tensor& input, std::span<const Axis> permutation)
	{
		return detail::apply_primitive(
			std::make_unique<detail::PermutePrimitive>(permutation, input.rank()), input);
	}

	Tensor transpose(const Tensor& input)
	{
		if (input.rank() < 2)
		{
			return input;
		}

		std::vector<Axis> permutation(input.rank());
		std::iota(permutation.begin(), permutation.end(), Axis{ 0 });
		std::reverse(permutation.begin(), permutation.end());
		return permute(input, permutation);
	}

	Tensor transpose(const Tensor& input, Axis axis0, Axis axis1)
	{
		const Shape::size_type normalized_axis0 = detail::normalize_axis(axis0, input.rank());
		const Shape::size_type normalized_axis1 = detail::normalize_axis(axis1, input.rank());
		if (normalized_axis0 == normalized_axis1)
		{
			return input;
		}

		std::vector<Axis> permutation(input.rank());
		std::iota(permutation.begin(), permutation.end(), Axis{ 0 });
		std::swap(permutation[normalized_axis0], permutation[normalized_axis1]);
		return permute(input, permutation);
	}

	Tensor reshape(const Tensor& input, Shape shape)
	{
		return detail::apply_primitive(
			std::make_unique<detail::ReshapePrimitive>(std::move(shape)), input);
	}

	Tensor flatten(const Tensor& input)
	{
		if (input.shape().is_scalar())
		{
			return reshape(input, Shape{ 1 });
		}
		return flatten(input, 0, -1);
	}

	Tensor flatten(const Tensor& input, Axis start_axis, Axis end_axis)
	{
		const Shape::size_type normalized_start = detail::normalize_axis(start_axis, input.rank());
		const Shape::size_type normalized_end = detail::normalize_axis(end_axis, input.rank());
		if (normalized_start > normalized_end)
		{
			throw std::invalid_argument{ "flatten start axis must not follow the end axis" };
		}
		if (normalized_start == normalized_end)
		{
			return input;
		}
		const auto dimensions = input.shape().dimensions();
		std::vector<Extent> output_dimensions;
		output_dimensions.reserve(input.rank() - (normalized_end - normalized_start));
		output_dimensions.insert(
			output_dimensions.end(), dimensions.begin(), dimensions.begin() + normalized_start);
		output_dimensions.push_back(flattened_extent(
			input.shape(), normalized_start, normalized_end));
		output_dimensions.insert(
			output_dimensions.end(), dimensions.begin() + normalized_end + 1, dimensions.end());
		return reshape(input, Shape{ std::move(output_dimensions) });
	}

	Tensor squeeze(const Tensor& input)
	{
		std::vector<Extent> output_dimensions;
		output_dimensions.reserve(input.rank());
		for (const Extent extent : input.shape().dimensions())
		{
			if (extent != 1)
			{
				output_dimensions.push_back(extent);
			}
		}

		Shape output_shape{ std::move(output_dimensions) };
		if (output_shape == input.shape())
		{
			return input;
		}
		return reshape(input, std::move(output_shape));
	}

	Tensor squeeze(const Tensor& input, Axis axis)
	{
		const std::array<Axis, 1> axes{ axis };
		return squeeze(input, axes);
	}

	Tensor squeeze(const Tensor& input, std::span<const Axis> axes)
	{
		if (axes.empty())
		{
			return input;
		}

		std::vector<bool> removed_axes(input.rank(), false);
		for (const Axis axis : axes)
		{
			const Shape::size_type normalized = detail::normalize_axis(axis, input.rank());
			if (removed_axes[normalized])
			{
				throw std::invalid_argument{ "squeeze axes must not contain duplicates" };
			}
			if (input.shape()[normalized] != 1)
			{
				throw std::invalid_argument{
					"squeeze requires every selected axis to have extent 1" };
			}
			removed_axes[normalized] = true;
		}

		std::vector<Extent> output_dimensions;
		output_dimensions.reserve(input.rank() - axes.size());
		for (Shape::size_type axis = 0; axis < input.rank(); ++axis)
		{
			if (!removed_axes[axis])
			{
				output_dimensions.push_back(input.shape()[axis]);
			}
		}
		return reshape(input, Shape{ std::move(output_dimensions) });
	}

	Tensor unsqueeze(const Tensor& input, Axis axis)
	{
		const Shape::size_type output_rank = input.rank() + 1;
		const Shape::size_type normalized = detail::normalize_axis(axis, output_rank);
		const auto dimensions = input.shape().dimensions();

		std::vector<Extent> output_dimensions;
		output_dimensions.reserve(output_rank);
		output_dimensions.insert(
			output_dimensions.end(), dimensions.begin(), dimensions.begin() + normalized);
		output_dimensions.push_back(1);
		output_dimensions.insert(
			output_dimensions.end(), dimensions.begin() + normalized, dimensions.end());
		return reshape(input, Shape{ std::move(output_dimensions) });
	}

	Tensor broadcast_to(const Tensor& input, Shape shape)
	{
		return detail::apply_primitive(
			std::make_unique<detail::BroadcastToPrimitive>(std::move(shape)), input);
	}

	Tensor concatenate(std::span<const Tensor> inputs, Axis axis)
	{
		if (inputs.empty())
		{
			throw std::invalid_argument{ "concatenate requires at least one input tensor" };
		}
		if (inputs.size() == 1)
		{
			return inputs.front();
		}
		return detail::apply_primitive(
			std::make_unique<detail::ConcatenatePrimitive>(axis, inputs.front().rank()), inputs);
	}

	Tensor slice(
		const Tensor& input,
		Axis axis,
		std::optional<Extent> start,
		std::optional<Extent> stop,
		Extent step)
	{
		return detail::apply_primitive(
			std::make_unique<detail::SlicePrimitive>(
				axis, start, stop, step, input.shape()),
			input);
	}
}
