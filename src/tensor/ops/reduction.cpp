#include <minitensor/ops/reduction.hpp>

#include <initializer_list>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include <minitensor/ops/elementwise.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "apply_primitive.hpp"
#include "tensor/core/axis.hpp"
#include "tensor/primitives/reduction/logsumexp.hpp"
#include "tensor/primitives/reduction/sum.hpp"

namespace minitensor
{

	// This helper is used by the composed mean operation.
	// It currently repeats reduction logic in the operation layer,
	// but this will be resolved once compiler IR is considered in the future
	namespace
	{
		Shape::size_type reduction_element_count(
			const Shape& input_shape,
			std::span<const Axis> axes)
		{
			std::vector<Extent> reduced_extents;
			reduced_extents.reserve(axes.size());
			for (const Axis axis : axes)
			{
				reduced_extents.push_back(
					input_shape[detail::normalize_axis(axis, input_shape.rank())]);
			}
			return Shape{ std::move(reduced_extents) }.numel();
		}
	}

	Tensor sum(const Tensor& input, std::span<const Axis> axes, bool keep_dim)
	{
		if (axes.empty())
		{
			return input;
		}
		return detail::apply_primitive(
			std::make_unique<detail::SumPrimitive>(axes, input.rank(), keep_dim), input);
	}

	Tensor sum(const Tensor& input, std::initializer_list<Axis> axes, bool keep_dim)
	{
		return sum(input, std::span<const Axis>(axes.begin(), axes.size()), keep_dim);
	}

	Tensor sum(const Tensor& input, bool keep_dim)
	{
		std::vector<Axis> axes(input.rank());
		std::iota(axes.begin(), axes.end(), Axis{ 0 });
		return sum(input, axes, keep_dim);
	}

	Tensor mean(const Tensor& input, std::span<const Axis> axes, bool keep_dim)
	{
		if (axes.empty())
		{
			return input;
		}

		const Tensor total = sum(input, axes, keep_dim);
		const Shape::size_type reduction_size = reduction_element_count(input.shape(), axes);
		return total / static_cast<float>(reduction_size);
	}

	Tensor mean(const Tensor& input, std::initializer_list<Axis> axes, bool keep_dim)
	{
		return mean(input, std::span<const Axis>(axes.begin(), axes.size()), keep_dim);
	}

	Tensor mean(const Tensor& input, bool keep_dim)
	{
		std::vector<Axis> axes(input.rank());
		std::iota(axes.begin(), axes.end(), Axis{ 0 });
		return mean(input, axes, keep_dim);
	}

	Tensor logsumexp(const Tensor& input, std::span<const Axis> axes, bool keep_dim)
	{
		if (axes.empty())
		{
			return input;
		}
		return detail::apply_primitive(
			std::make_unique<detail::LogSumExpPrimitive>(axes, input.rank(), keep_dim), input);
	}

	Tensor logsumexp(const Tensor& input, std::initializer_list<Axis> axes, bool keep_dim)
	{
		return logsumexp(input, std::span<const Axis>(axes.begin(), axes.size()), keep_dim);
	}

	Tensor logsumexp(const Tensor& input, bool keep_dim)
	{
		std::vector<Axis> axes(input.rank());
		std::iota(axes.begin(), axes.end(), Axis{ 0 });
		return logsumexp(input, axes, keep_dim);
	}

}
