#pragma once

#include <cstdint>
#include <span>

#include <minitensor/types.hpp>

#include "generator_registry.hpp"
#include "runtime_registry.hpp"
#include "tensor/core/random.hpp"
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
		[[nodiscard]] RandomKey reserve_random_key(const Device& device);
		void manual_seed(std::uint64_t seed);

	private:
		RuntimeRegistry runtimes_;
		KernelRegistry kernels_;
		GeneratorRegistry generators_;
	};

	[[nodiscard]] ExecutionEnvironment& environment();
}
