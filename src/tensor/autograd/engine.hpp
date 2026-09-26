#pragma once

#include <span>
#include <vector>

#include <minitensor/tensor.hpp>

#include "tensor/graph/fwd.hpp"

namespace minitensor::detail
{
	[[nodiscard]] std::vector<Tensor> reverse_vjp(const ValueRef& output, std::span<const ValueRef> targets, const Tensor& output_cotangent);
}