#include <minitensor/data.hpp>

#include <algorithm>
#include <cstdint>
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
	template <TensorElement T>
	Tensor from_data(std::span<const T> data, Shape shape, Device device)
	{
		if (data.size() != shape.numel())
		{
			throw std::invalid_argument{ "data size does not match requested shape" };
		}

		detail::TensorSpec spec{ std::move(shape), dtype_of_v<T>, device };
		detail::ValueRef value = detail::make_value_from_host(std::move(spec), std::as_bytes(data));
		return detail::TensorAccess::make(std::move(value));
	}

	template <TensorElement T>
	std::vector<T> to_vector(const Tensor& tensor)
	{
		if (tensor.dtype() != dtype_of_v<T>)
		{
			throw std::invalid_argument{ "requested host type does not match tensor dtype" };
		}

		const Tensor contiguous_tensor = contiguous(tensor);
		eval(contiguous_tensor);
		const detail::ValueRef& value = detail::TensorAccess::value(contiguous_tensor);
		std::vector<T> result(tensor.numel());
		detail::copy_value_to_host(std::as_writable_bytes(std::span<T>(result)), value);
		return result;
	}

	template <TensorElement T>
	T item(const Tensor& tensor)
	{
		if (tensor.numel() != 1)
		{
			throw std::invalid_argument{ "item() requires a single tensor element" };
		}
		return to_vector<T>(tensor).front();
	}

	// Templates are instantiated explicitely
	template std::vector<float> to_vector<float>(const Tensor& tensor);
	template std::vector<std::int64_t> to_vector<std::int64_t>(const Tensor& tensor);
	template float item<float>(const Tensor& tensor);
	template std::int64_t item<std::int64_t>(const Tensor& tensor);
	template Tensor from_data<float>(
		std::span<const float> data,
		Shape shape,
		Device device);
	template Tensor from_data<std::int64_t>(
		std::span<const std::int64_t> data,
		Shape shape,
		Device device);
}
