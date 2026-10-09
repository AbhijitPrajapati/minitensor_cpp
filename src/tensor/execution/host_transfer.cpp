#include "host_transfer.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>

#include <minitensor/types.hpp>

#include "environment.hpp"
#include "tensor/backend/device_runtime.hpp"
#include "tensor/core/dense_size.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/leaf.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/storage/buffer.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::detail
{
	ValueRef make_value_from_host(TensorSpec spec, std::span<const std::byte> source)
	{
		const std::size_t expected_size = dense_size_bytes(spec);
		if (source.size() != expected_size)
		{
			throw std::invalid_argument{ "host data size does not match value specification" };
		}

		DeviceRuntime& runtime = environment().runtime_for(spec.device);
		BufferRef buffer = runtime.allocate(expected_size);
		if (!buffer)
		{
			throw std::runtime_error{ "null buffer allocated" };
		}

		runtime.copy_from_host(*buffer, 0, source);
		Materialization materialization(std::move(buffer), Layout::contiguous(spec.shape));
		return make_materialized_leaf(std::move(spec), std::move(materialization));
	}

	void copy_value_to_host(std::span<std::byte> destination, const ValueRef& value)
	{
		const Materialization* materialization = value->materialization();
		if (materialization == nullptr)
		{
			throw std::logic_error{ "cannot copy unmaterialized value to host" };
		}

		const TensorSpec& spec = value->spec();
		if (destination.size() != dense_size_bytes(spec))
		{
			throw std::invalid_argument{ "host destination size does not match value specification" };
		}

		if (destination.empty())
		{
			return;
		}

		const Layout& layout = materialization->layout();
		if (!layout.is_contiguous(spec.shape))
		{
			throw std::logic_error{ "value must be contiguous for host copy" };
		}

		if (layout.offset() < 0)
		{
			throw std::logic_error{ "contiguous value has negative storage offset" };
		}

		const auto source_offset_bytes =
			static_cast<std::size_t>(layout.offset()) * dtype_size(spec.dtype);

		DeviceRuntime& runtime = environment().runtime_for(spec.device);
		runtime.copy_to_host(destination, *materialization->buffer_ref(), source_offset_bytes);
	}
}