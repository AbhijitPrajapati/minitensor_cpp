#pragma once

#include <span>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor
{
	[[nodiscard]] Tensor sum(const Tensor& input, std::span<const Axis> axes, bool keep_dim = false);
	[[nodiscard]] Tensor sum(const Tensor& input, bool keep_dim = false);
}
