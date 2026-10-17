#include "basic_index.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "tensor/core/axis.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/apply_operation.hpp"
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

		Extent normalize_index(Extent index, Extent axis_extent)
		{
			if (index < 0)
			{
				index += axis_extent;
			}
			if (index < 0 || index >= axis_extent)
			{
				throw std::out_of_range{ "select index is out of range" };
			}
			return index;
		}
	}

	const Shape& BasicIndexParameters::input_shape() const noexcept
	{
		return input_shape_;
	}

	const Shape& BasicIndexParameters::output_shape() const noexcept
	{
		return output_shape_;
	}

	BasicIndexParameters::BasicIndexParameters(Shape input_shape, Shape output_shape, std::vector<Component> components)
		: input_shape_(std::move(input_shape)), output_shape_(std::move(output_shape)), components_(std::move(components))
	{}

	// Parameter construction for slicing
	BasicIndexParameters BasicIndexParameters::for_slice(
		Axis axis,
		std::optional<Extent> start,
		std::optional<Extent> stop,
		Extent step,
		const Shape& input_shape)
	{
		const NormalizedSlice normalized = normalize_slice(axis, start, stop, step, input_shape);

		std::vector<Component> components(input_shape.rank(), AllIndex{});
		components[normalized.axis] = SliceIndex{ normalized.start, normalized.length, normalized.step };

		std::vector<Extent> output_extents(
			input_shape.dimensions().begin(), input_shape.dimensions().end());
		output_extents[normalized.axis] = normalized.length;

		return BasicIndexParameters(input_shape, Shape{ std::move(output_extents) }, std::move(components));
	}

	// Parameter construction for selection
	BasicIndexParameters BasicIndexParameters::for_select(Axis axis, Extent index, const Shape& input_shape)
	{
		const Shape::size_type normalized_axis = normalize_axis(axis, input_shape.rank());
		index = normalize_index(index, input_shape[normalized_axis]);

		const auto input_extents = input_shape.dimensions();
		std::vector<Extent> output_extents;
		output_extents.reserve(input_shape.rank() - 1);
		output_extents.insert(
			output_extents.end(),
			input_extents.begin(),
			input_extents.begin() + normalized_axis);
		output_extents.insert(
			output_extents.end(),
			input_extents.begin() + normalized_axis + 1,
			input_extents.end());

		std::vector<Component> components(input_shape.rank(), AllIndex{});
		components[normalized_axis] = SelectIndex{ index };

		return BasicIndexParameters(input_shape, Shape{ std::move(output_extents) }, std::move(components));
	}

	// Layout derivation is determined by index parameters, so it is delegated here.
	Layout BasicIndexParameters::derive_layout(const Layout& input_layout) const
	{
		std::vector<Layout::stride_type> output_strides;
		output_strides.reserve(output_shape_.rank());
		Layout::offset_type output_offset = input_layout.offset();
		const bool output_has_elements = output_shape_.numel() != 0;

		for (Shape::size_type axis = 0; axis < input_shape_.rank(); ++axis)
		{
			const Layout::stride_type input_stride = input_layout.stride(axis);
			std::visit(
				[&](const auto& index)
				{
					using Index = std::remove_cvref_t<decltype(index)>;
					if constexpr (std::is_same_v<Index, AllIndex>)
					{
						output_strides.push_back(input_stride);
					}
					else if constexpr (std::is_same_v<Index, SliceIndex>)
					{
						if (output_has_elements)
						{
							output_offset += input_stride * index.start;
						}
						output_strides.push_back(
							index.length > 1
								? input_stride * index.step
								: input_stride);
					}
					else
					{
						if (output_has_elements)
						{
							output_offset += input_stride * index.index;
						}
					}
				},
				components_[axis]);
		}

		return Layout{ std::move(output_strides), output_offset };
	}

	BasicIndexPrimitive::BasicIndexPrimitive(BasicIndexParameters parameters)
		: parameters_(std::move(parameters))
	{}

	std::string_view BasicIndexPrimitive::name() const noexcept
	{
		return "basic_index";
	}

	bool BasicIndexPrimitive::requires_kernel_support() const noexcept
	{
		return false;
	}

	TensorSpec BasicIndexPrimitive::infer(std::span<const TensorSpec> inputs) const
	{
		if (inputs.size() != 1)
		{
			throw std::invalid_argument{ "basic index requires a single input" };
		}

		const TensorSpec& input = inputs.front();
		if (input.shape != parameters_.input_shape())
		{
			throw std::invalid_argument{ "basic index primitive was normalized for a different shape" };
		}

		return TensorSpec{ parameters_.output_shape(), input.dtype, input.device };
	}

	std::optional<Layout> BasicIndexPrimitive::try_derive_shared_layout(const TensorSpec&, const Layout& input_layout, const TensorSpec&) const
	{
		return parameters_.derive_layout(input_layout);
	}

	std::vector<std::optional<Tensor>> BasicIndexPrimitive::vjp(std::span<const Tensor> inputs, const Tensor&, const Tensor& output_cotangent) const
	{
		if (inputs.size() != 1)
		{
			throw std::logic_error{ "basic index primitive VJP expects 1 input" };
		}

		return { apply_operation(
			std::make_unique<BasicIndexScatterPrimitive>(parameters_),
			output_cotangent) };
	}

	BasicIndexScatterPrimitive::BasicIndexScatterPrimitive(BasicIndexParameters parameters)
		: parameters_(std::move(parameters))
	{}

	std::string_view BasicIndexScatterPrimitive::name() const noexcept
	{
		return "basic_index_scatter";
	}

	TensorSpec BasicIndexScatterPrimitive::infer(
		std::span<const TensorSpec> inputs) const
	{
		if (inputs.size() != 1)
		{
			throw std::invalid_argument{ "basic index scatter requires a single input" };
		}

		const TensorSpec& input = inputs.front();
		if (input.shape != parameters_.output_shape())
		{
			throw std::invalid_argument{
				"basic index scatter input shape does not match the indexed shape" };
		}
		return TensorSpec{ parameters_.input_shape(), input.dtype, input.device };
	}

	const BasicIndexParameters& BasicIndexScatterPrimitive::parameters() const noexcept
	{
		return parameters_;
	}

	std::vector<std::optional<Tensor>> BasicIndexScatterPrimitive::vjp(
		std::span<const Tensor> inputs,
		const Tensor&,
		const Tensor& output_cotangent) const
	{
		if (inputs.size() != 1)
		{
			throw std::logic_error{ "basic index scatter VJP expects 1 input" };
		}

		return { apply_operation(
			std::make_unique<BasicIndexPrimitive>(parameters_),
			output_cotangent) };
	}
}
