#include "logsumexp.hpp"

#include <algorithm>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <minitensor/ops/elementwise.hpp>
#include <minitensor/ops/manipulation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "tensor/core/axis.hpp"
#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
	LogSumExpPrimitive::LogSumExpPrimitive(std::span<const Axis> axes, Shape::size_type input_rank, bool keep_dim)
		: input_rank_(input_rank), keep_dim_(keep_dim)
	{
		if (axes.size() > input_rank_)
		{
			throw std::invalid_argument{
				"log sum exp cannot reduce more unique axes than the input rank" };
		}

		axes_.reserve(axes.size());
		for (const Axis axis : axes)
		{
			axes_.push_back(normalize_axis(axis, input_rank_));
		}
		std::ranges::sort(axes_);
		if (std::ranges::adjacent_find(axes_) != axes_.end())
		{
			throw std::invalid_argument{ "log sum exp axes cannot contain duplicates" };
		}
	}

	std::string_view LogSumExpPrimitive::name() const noexcept
	{
		return "log_sum_exp";
	}

	TensorSpec LogSumExpPrimitive::infer(std::span<const TensorSpec> inputs) const
	{
		if (inputs.size() != 1)
		{
			throw std::invalid_argument{ "log sum exp requires a single input" };
		}

		const TensorSpec& input = inputs.front();
		if (input.shape.rank() != input_rank_)
		{
			throw std::invalid_argument{
				"log sum exp primitive was normalized for a different input rank" };
		}
		if (!is_floating_point(input.dtype))
		{
			throw std::invalid_argument{ "log sum exp requires a floating-point input" };
		}

		std::vector<Extent> output_extents;
		if (keep_dim_)
		{
			const auto input_dimensions = input.shape.dimensions();
			output_extents.assign(input_dimensions.begin(), input_dimensions.end());
			for (const Shape::size_type axis : axes_)
			{
				output_extents[axis] = Extent{ 1 };
			}
		}
		else
		{
			output_extents.reserve(input.shape.rank() - axes_.size());
			auto reduced_axis = axes_.begin();
			for (Shape::size_type axis = 0; axis < input.shape.rank(); ++axis)
			{
				if (reduced_axis != axes_.end() && *reduced_axis == axis)
				{
					++reduced_axis;
					continue;
				}
				output_extents.push_back(input.shape[axis]);
			}
		}

		return TensorSpec{ Shape{std::move(output_extents)}, input.dtype, input.device };
	}

	const std::vector<Shape::size_type>& LogSumExpPrimitive::axes() const noexcept
	{
		return axes_;
	}

	bool LogSumExpPrimitive::keep_dim() const noexcept
	{
		return keep_dim_;
	}

	std::vector<std::optional<Tensor>> LogSumExpPrimitive::vjp(std::span<const Tensor> inputs, const Tensor& output, const Tensor& output_cotangent) const
	{
		if (inputs.size() != 1)
		{
			throw std::logic_error{ "log sum exp VJP expects 1 input" };
		}

		const Tensor& input = inputs.front();

		Tensor expanded_output = output;
		Tensor expanded_cotangent = output_cotangent;

		if (!keep_dim_)
		{
			std::vector<Extent> expanded_extents(
				input.shape().dimensions().begin(),
				input.shape().dimensions().end());

			for (const Shape::size_type axis : axes_)
			{
				expanded_extents[axis] = Extent{ 1 };
			}

			const Shape expanded_shape = Shape{ std::move(expanded_extents) };
			expanded_output = reshape(expanded_output, expanded_shape);
			expanded_cotangent = reshape(expanded_cotangent, expanded_shape);
		}

		return { expanded_cotangent * exp(input - expanded_output) };
	}
}
