#include <minitensor/data.hpp>

#include <initializer_list>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/evaluation.hpp>
#include <minitensor/ops/manipulation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/tensor_access.hpp"

#include "tensor/execution/host_transfer.hpp"

namespace minitensor
{
	Tensor from_data(std::span<const float> data, Shape shape, TensorOptions options)
	{
		if (data.size() != shape.numel())
		{
			throw std::invalid_argument{ "data size does not match requested shape" };
		}

		detail::TensorSpec spec{ std::move(shape), DType::Float32, options.device };
		detail::ValueRef value = detail::make_value_from_host(std::move(spec), std::as_bytes(data));
		return detail::TensorAccess::make(std::move(value));
	}

	Tensor from_data(std::initializer_list<float> data, Shape shape, TensorOptions options)
	{
		return from_data(std::span<const float>(data.begin(), data.size()), std::move(shape), options);
	}

	std::vector<float> to_vector(const Tensor& tensor)
	{
		if (tensor.dtype() != DType::Float32)
		{
			throw std::logic_error{ "unsupported datatype detected" };
		}

		const Tensor contiguous_tensor = contiguous(tensor);
		eval(contiguous_tensor);
		const detail::ValueRef& value = detail::TensorAccess::value(contiguous_tensor);
		std::vector<float> result(tensor.numel());
		detail::copy_value_to_host(std::as_writable_bytes(std::span<float>(result)), value);
		return result;
	}

	float item(const Tensor& tensor)
	{
		if (tensor.numel() != 1)
		{
			throw std::invalid_argument{ "item() requires a single tensor element" };
		}
		return to_vector(tensor).front();
	}
}
