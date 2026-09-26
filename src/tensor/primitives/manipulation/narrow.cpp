#include "narrow.hpp"

#include <array>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/manipulation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail
{
	namespace
	{
		[[nodiscard]] Layout::offset_type checked_slice_offset(
			Layout::offset_type offset,
			Layout::stride_type stride,
			Extent start)
		{
			constexpr auto min = std::numeric_limits<Layout::offset_type>::min();
			constexpr auto max = std::numeric_limits<Layout::offset_type>::max();

			Layout::offset_type contribution = 0;
			if (start != 0 && stride > 0 && stride > max / start)
			{
				throw std::overflow_error{ "narrow layout offset overflow" };
			}
			if (start != 0 && stride < 0 && stride < min / start)
			{
				throw std::overflow_error{ "narrow layout offset underflow" };
			}
			contribution = stride * start;

			if (contribution > 0 && offset > max - contribution)
			{
				throw std::overflow_error{ "narrow layout offset overflow" };
			}
			if (contribution < 0 && offset < min - contribution)
			{
				throw std::overflow_error{ "narrow layout offset underflow" };
			}
			return offset + contribution;
		}
	}

	NarrowPrimitive::NarrowPrimitive(
		Shape::size_type axis,
		Extent start,
		Extent length,
		Shape::size_type input_rank)
		: axis_(axis), start_(start), length_(length), input_rank_(input_rank)
	{
		if (axis_ >= input_rank_)
		{
			throw std::out_of_range{ "narrow axis is out of range for input rank" };
		}
		if (start_ < 0 || length_ < 0)
		{
			throw std::invalid_argument{ "narrow start and length must be nonnegative" };
		}
	}

	std::string_view NarrowPrimitive::name() const noexcept
	{
		return "narrow";
	}

	bool NarrowPrimitive::requires_kernel_support() const noexcept
	{
		return false;
	}

	TensorSpec NarrowPrimitive::infer(std::span<const TensorSpec> inputs) const
	{
		if (inputs.size() != 1)
		{
			throw std::invalid_argument{ "narrow requires a single input" };
		}

		const TensorSpec& input = inputs.front();
		if (input.shape.rank() != input_rank_)
		{
			throw std::invalid_argument{
				"narrow primitive was normalized for a different input rank" };
		}

		const Extent input_extent = input.shape[axis_];
		if (start_ > input_extent || length_ > input_extent - start_)
		{
			throw std::out_of_range{ "narrow range exceeds the input extent" };
		}

		std::vector<Extent> output_extents(
			input.shape.dimensions().begin(), input.shape.dimensions().end());
		output_extents[axis_] = length_;
		return TensorSpec{
			Shape{ std::move(output_extents) }, input.dtype, input.device };
	}

	std::optional<Layout> NarrowPrimitive::try_derive_shared_layout(
		const TensorSpec&,
		const Layout& input_layout,
		const TensorSpec& output_spec) const
	{
		std::vector<Layout::stride_type> output_strides(
			input_layout.strides().begin(), input_layout.strides().end());

		Layout::offset_type output_offset = input_layout.offset();
		if (output_spec.shape.numel() != 0)
		{
			output_offset = checked_slice_offset(
				output_offset, input_layout.stride(axis_), start_);
		}
		return Layout{ std::move(output_strides), output_offset };
	}

	std::vector<std::optional<Tensor>> NarrowPrimitive::vjp(
		std::span<const Tensor> inputs,
		const Tensor&,
		const Tensor& output_cotangent) const
	{
		if (inputs.size() != 1)
		{
			throw std::logic_error{ "narrow VJP expects 1 input" };
		}

		const Tensor& input = inputs.front();
		const Extent suffix_length = input.shape()[axis_] - start_ - length_;
		std::vector<Tensor> sections;
		sections.reserve(3);

		auto zero_section = [&](Extent axis_extent)
			{
				std::vector<Extent> dimensions(
					input.shape().dimensions().begin(), input.shape().dimensions().end());
				dimensions[axis_] = axis_extent;
				return zeros(
					Shape{ std::move(dimensions) },
					TensorOptions{ input.dtype(), input.device() });
			};

		if (start_ != 0)
		{
			sections.push_back(zero_section(start_));
		}
		sections.push_back(output_cotangent);
		if (suffix_length != 0)
		{
			sections.push_back(zero_section(suffix_length));
		}

		return { concatenate(sections, static_cast<Axis>(axis_)) };
	}
}
