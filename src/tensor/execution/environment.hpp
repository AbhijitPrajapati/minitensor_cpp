#pragma once

#include <span>

#include <minitensor/types.hpp>

#include "runtime_registry.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/graph/fwd.hpp"

namespace minitensor::detail
{
	class DeviceRuntime;

	class ExecutionEnvironment final
	{
	public:
		ExecutionEnvironment();
		void evaluate(std::span<const ValueRef> roots);
		DeviceRuntime& runtime_for(const Device& device);

	private:
		RuntimeRegistry runtimes_;
		KernelRegistry kernels_;
	};

	[[nodiscard]] ExecutionEnvironment& environment();
}
