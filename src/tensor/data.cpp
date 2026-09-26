#include <minitensor/data.hpp>

#include <cassert>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/evaluation.hpp>
#include <minitensor/ops/manipulation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "tensor/backend/device_runtime.hpp"
#include "tensor/core/dense_size.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/execution/environment.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/storage/buffer.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"
#include "tensor/tensor_access.hpp"

namespace minitensor
{
	Tensor from_data(std::span<const float> data, Shape shape, TensorOptions options)
	{
		if (data.size() != shape.numel())
		{
			throw std::invalid_argument{ "data size does not match requested shape" };
		}

		detail::TensorSpec spec{ std::move(shape), DType::Float32, options.device };
		detail::DeviceRuntime& runtime = detail::environment().runtime_for(spec.device);
		detail::BufferRef buffer = runtime.allocate(detail::dense_size_bytes(spec));
		if (!buffer)
		{
			throw std::runtime_error{ "null buffer allocated" };
		}
		runtime.copy_from_host(*buffer, 0, std::as_bytes(data));
		detail::Materialization materialization(std::move(buffer), detail::Layout::contiguous(spec.shape));
		auto value = std::make_shared<detail::Value>(std::move(spec));
		value->materialize(std::move(materialization));
		return detail::TensorAccess::make(std::move(value));
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
		const detail::Materialization* materialization = value->materialization();
		if (!materialization)
		{
			throw std::logic_error{ "value is unmaterialized after evaluation" };
		}

		const detail::TensorSpec& spec = value->spec();
		std::vector<float> result(spec.shape.numel());
		if (result.empty())
		{
			return result;
		}

		detail::DeviceRuntime& runtime = detail::environment().runtime_for(spec.device);
		const detail::BufferRef& buffer = materialization->buffer_ref();
		const detail::Layout& layout = materialization->layout();
		assert(layout.is_contiguous(spec.shape));
		assert(layout.offset() >= 0);
		const auto source_offset_bytes =
			static_cast<std::size_t>(layout.offset()) * sizeof(float);
		runtime.copy_to_host(
			std::as_writable_bytes(std::span<float>(result)),
			*buffer,
			source_offset_bytes);

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
