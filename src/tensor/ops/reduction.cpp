#include <minitensor/ops/reduction.hpp>

#include <initializer_list>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "apply_primitive.hpp"
#include "tensor/primitives/reduction/sum.hpp"

namespace minitensor
{
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

}
