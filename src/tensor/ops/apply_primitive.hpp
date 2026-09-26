#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <utility>
#include <vector>

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

	[[nodiscard]] inline Tensor apply_primitive(
		std::unique_ptr<Primitive> primitive,
		std::span<const Tensor> inputs)
	{
		std::vector<ValueRef> input_values;
		input_values.reserve(inputs.size());
		for (const Tensor& input : inputs)
		{
			input_values.push_back(TensorAccess::value(input));
		}

		ValueRef output = apply_operation(std::move(primitive), input_values);
		return TensorAccess::make(std::move(output));
	}
}
