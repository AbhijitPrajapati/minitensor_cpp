#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <utility>

#include <minitensor/tensor.hpp>

#include "tensor/graph/apply_operation.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/tensor_access.hpp"

namespace minitensor::detail
{
	template <typename... Inputs>
	[[nodiscard]] Tensor apply_primitive(
		std::unique_ptr<Primitive> primitive,
		const Inputs &...inputs)
	{
		std::array<ValueRef, sizeof...(Inputs)> input_values{
			TensorAccess::value(inputs)... };
		ValueRef output = apply_operation(std::move(primitive), input_values);
		return TensorAccess::make(std::move(output));
	}
}
