#include "apply_operation.hpp"

#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <minitensor/tensor.hpp>

#include "graph/apply_primitive.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/tensor_access.hpp"

namespace minitensor::detail
{
	Tensor apply_operation(
		std::unique_ptr<Primitive> primitive,
		std::span<const Tensor> inputs)
	{
		std::vector<ValueRef> input_values;
		input_values.reserve(inputs.size());

		for (const Tensor& input : inputs)
		{
			input_values.push_back(TensorAccess::value(input));
		}

		return TensorAccess::make(
			apply_primitive(std::move(primitive), input_values));
	}
}
