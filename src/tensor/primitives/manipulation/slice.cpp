#include "slice.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "tensor/core/axis.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/ops/apply_primitive.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail
{
	namespace
	{
		struct NormalizedSlice final
		{
			Shape::size_type axis;
			Extent start;
			Extent length;
			Extent step;
		};

		Extent normalize_bound(
			Extent bound,
			Extent axis_extent,
			bool reverse)
		{
			if (bound < 0)
			{
				if (bound < -axis_extent)
				{
					return reverse ? Extent{ -1 } : Extent{ 0 };
				}
				return bound + axis_extent;
			}

			if (bound >= axis_extent)
			{
				return reverse ? axis_extent - 1 : axis_extent;
			}
			return bound;
		}

		Extent slice_length(
			Extent start,
			Extent stop,
			Extent step) noexcept
		{
			if (step > 0)
			{
				if (start >= stop)
				{
					return 0;
				}
				return 1 + (stop - 1 - start) / step;
			}

			if (start <= stop)
			{
				return 0;
			}

			const auto distance = static_cast<std::uint64_t>(start - 1 - stop);
			const auto step_magnitude =
				static_cast<std::uint64_t>(-(step + 1)) + std::uint64_t{ 1 };
			return static_cast<Extent>(
				std::uint64_t{ 1 } + distance / step_magnitude);
		}

		NormalizedSlice normalize_slice(
			Axis axis,
			std::optional<Extent> start,
			std::optional<Extent> stop,
			Extent step,
			const Shape& input_shape)
		{
			if (step == 0)
			{
				throw std::invalid_argument{ "slice step cannot be zero" };
			}

			const Shape::size_type normalized_axis =
				normalize_axis(axis, input_shape.rank());
			const Extent axis_extent = input_shape[normalized_axis];
			const bool reverse = step < 0;

			const Extent normalized_start = start.has_value()
				? normalize_bound(*start, axis_extent, reverse)
				: (reverse ? axis_extent - 1 : Extent{ 0 });
			const Extent normalized_stop = stop.has_value()
				? normalize_bound(*stop, axis_extent, reverse)
				: (reverse ? Extent{ -1 } : axis_extent);

			return NormalizedSlice{
				normalized_axis,
				normalized_start,
				slice_length(normalized_start, normalized_stop, step),
				step };
		}
	}

	SliceParameters::SliceParameters(
		Axis axis,
		std::optional<Extent> start,
		std::optional<Extent> stop,
		Extent step,
		const Shape& input_shape)
	{
		const NormalizedSlice normalized = normalize_slice(axis, start, stop, step, input_shape);
		axis_ = normalized.axis;
		start_ = normalized.start;
		length_ = normalized.length;
		step_ = normalized.step;
		input_rank_ = input_shape.rank();
		input_axis_extent_ = input_shape[axis_];
	}

	Shape::size_type SliceParameters::axis() const noexcept
	{
		return axis_;
	}

	Extent SliceParameters::start() const noexcept
	{
		return start_;
	}

	Extent SliceParameters::length() const noexcept
	{
		return length_;
	}

	Extent SliceParameters::step() const noexcept
	{
		return step_;
	}

	Shape::size_type SliceParameters::input_rank() const noexcept
	{
		return input_rank_;
	}

	Extent SliceParameters::input_axis_extent() const noexcept
	{
		return input_axis_extent_;
	}

	SlicePrimitive::SlicePrimitive(
		Axis axis,
		std::optional<Extent> start,
		std::optional<Extent> stop,
		Extent step,
		const Shape& input_shape)
		: parameters_(axis, start, stop, step, input_shape)
	{}

	SlicePrimitive::SlicePrimitive(SliceParameters parameters)
		: parameters_(std::move(parameters))
	{}

	std::string_view SlicePrimitive::name() const noexcept
	{
		return "slice";
	}

	bool SlicePrimitive::requires_kernel_support() const noexcept
	{
		return false;
	}

	TensorSpec SlicePrimitive::infer(std::span<const TensorSpec> inputs) const
	{
		if (inputs.size() != 1)
		{
			throw std::invalid_argument{ "slice requires a single input" };
		}

		const TensorSpec& input = inputs.front();
		if (input.shape.rank() != parameters_.input_rank())
		{
			throw std::invalid_argument{
				"slice primitive was normalized for a different input rank" };
		}
		if (input.shape[parameters_.axis()] != parameters_.input_axis_extent())
		{
			throw std::invalid_argument{
				"slice primitive was normalized for a different input axis extent" };
		}

		std::vector<Extent> output_extents(
			input.shape.dimensions().begin(), input.shape.dimensions().end());
		output_extents[parameters_.axis()] = parameters_.length();
		return TensorSpec{
			Shape{ std::move(output_extents) }, input.dtype, input.device };
	}

	std::optional<Layout> SlicePrimitive::try_derive_shared_layout(
		const TensorSpec&,
		const Layout& input_layout,
		const TensorSpec& output_spec) const
	{
		std::vector<Layout::stride_type> output_strides(
			input_layout.strides().begin(), input_layout.strides().end());
		Layout::offset_type output_offset = input_layout.offset();

		if (output_spec.shape.numel() != 0)
		{
			const Layout::stride_type input_stride =
				input_layout.stride(parameters_.axis());
			output_offset += input_stride * parameters_.start();

			// A singleton dimension never advances, so its stride is immaterial.
			if (parameters_.length() > 1)
			{
				output_strides[parameters_.axis()] = input_stride * parameters_.step();
			}
		}

		return Layout{ std::move(output_strides), output_offset };
	}

	std::vector<std::optional<Tensor>> SlicePrimitive::vjp(
		std::span<const Tensor> inputs,
		const Tensor&,
		const Tensor& output_cotangent) const
	{
		if (inputs.size() != 1)
		{
			throw std::logic_error{ "slice VJP expects 1 input" };
		}

		return { apply_primitive(
			std::make_unique<SliceScatterPrimitive>(
				inputs.front().shape(), parameters_),
			output_cotangent) };
	}

	SliceScatterPrimitive::SliceScatterPrimitive(
		Shape output_shape,
		SliceParameters parameters)
		: output_shape_(std::move(output_shape)),
		parameters_(std::move(parameters))
	{
		if (output_shape_.rank() != parameters_.input_rank() ||
			output_shape_[parameters_.axis()] != parameters_.input_axis_extent())
		{
			throw std::invalid_argument{
				"slice_scatter output shape does not match the normalized slice" };
		}

		std::vector<Extent> input_extents(
			output_shape_.dimensions().begin(), output_shape_.dimensions().end());
		input_extents[parameters_.axis()] = parameters_.length();
		input_shape_ = Shape{ std::move(input_extents) };
	}

	std::string_view SliceScatterPrimitive::name() const noexcept
	{
		return "slice_scatter";
	}

	TensorSpec SliceScatterPrimitive::infer(
		std::span<const TensorSpec> inputs) const
	{
		if (inputs.size() != 1)
		{
			throw std::invalid_argument{ "slice_scatter requires a single input" };
		}

		const TensorSpec& input = inputs.front();
		if (input.shape != input_shape_)
		{
			throw std::invalid_argument{
				"slice_scatter input shape does not match the sliced shape" };
		}
		return TensorSpec{ output_shape_, input.dtype, input.device };
	}

	Shape::size_type SliceScatterPrimitive::axis() const noexcept
	{
		return parameters_.axis();
	}

	Extent SliceScatterPrimitive::start() const noexcept
	{
		return parameters_.start();
	}

	Extent SliceScatterPrimitive::step() const noexcept
	{
		return parameters_.step();
	}

	std::vector<std::optional<Tensor>> SliceScatterPrimitive::vjp(
		std::span<const Tensor> inputs,
		const Tensor&,
		const Tensor& output_cotangent) const
	{
		if (inputs.size() != 1)
		{
			throw std::logic_error{ "slice_scatter VJP expects 1 input" };
		}

		return { apply_primitive(
			std::make_unique<SlicePrimitive>(parameters_),
			output_cotangent) };
	}
}
