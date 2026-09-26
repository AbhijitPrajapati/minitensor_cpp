#include <minitensor/autograd.hpp>

#include <span>
#include <stdexcept>
#include <vector>

#include <minitensor/ops/creation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "tensor/autograd/engine.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/tensor_access.hpp"

namespace minitensor
{
	std::vector<Tensor> vjp(const Tensor& output, std::span<const Tensor> inputs, const Tensor& output_cotangent)
	{
		std::vector<detail::ValueRef> targets;
		targets.reserve(inputs.size());
		for (const Tensor& input : inputs)
		{
			targets.push_back(detail::TensorAccess::value(input));
		}
		return detail::reverse_vjp(detail::TensorAccess::value(output), targets, output_cotangent);
	}

	std::vector<Tensor> grad(const Tensor& output, std::span<const Tensor> inputs)
	{
		if (output.rank() != 0)
		{
			throw std::invalid_argument{ "grad requires rank 0 output" };
		}
		Tensor seed = full(Shape{}, 1.0F, TensorOptions{ output.dtype(), output.device() });
		return vjp(output, inputs, seed);
	}
}
