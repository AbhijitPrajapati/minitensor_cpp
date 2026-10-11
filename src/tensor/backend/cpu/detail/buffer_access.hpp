#pragma once

#include <cassert>
#include <minitensor/types.hpp>

#include "tensor/backend/cpu/cpu_buffer.hpp"
#include "tensor/dispatch/tensor_view.hpp"

namespace minitensor::detail::cpu
{
	template <typename T>
	concept CpuElement = TensorElement<T>;

	// Accessors for storage buffers
	
	template <CpuElement T>
	[[nodiscard]] const T* data(const TensorView& view)
	{
		assert(view.dtype() == dtype_of_v<T>);
		const auto& buffer = dynamic_cast<const CpuBuffer&>(view.buffer());
		return reinterpret_cast<const T*>(buffer.data());
	}

	template <CpuElement T>
	[[nodiscard]] T* data(const MutableTensorView& view)
	{
		assert(view.dtype() == dtype_of_v<T>);
		auto& buffer = dynamic_cast<CpuBuffer&>(view.buffer());
		return reinterpret_cast<T*>(buffer.data());
	}
}
