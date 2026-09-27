#include <minitensor/types.hpp>

#include <span>
#include <stdexcept>
#include <typeinfo>
#include <utility>

#include <gtest/gtest.h>

#include "tensor/backend/cpu/cpu_runtime.hpp"
#include "tensor/backend/device_runtime.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"
#include "tensor/support/test_buffer.hpp"
#include "tensor/support/test_primitive.hpp"

namespace minitensor::test
{
	TEST(KernelKeyTest, IncludesPrimitiveTypeAndDeviceType)
	{
		const detail::KernelKey key{ typeid(IdentitySpecPrimitive), DeviceType::Cpu };
		const detail::KernelKey equal{ typeid(IdentitySpecPrimitive), DeviceType::Cpu };
		const detail::KernelKey other{ typeid(DestructionTrackedPrimitive), DeviceType::Cpu };
		EXPECT_EQ(key, equal);
		EXPECT_NE(key, other);
	}

	TEST(KernelRegistryTest, RegistersAndRetrievesKernels)
	{
		detail::KernelRegistry registry;
		const detail::KernelKey key{ typeid(IdentitySpecPrimitive), DeviceType::Cpu };
		int calls = 0;
		registry.register_kernel(
			key,
			[&calls](
				detail::DeviceRuntime&,
				const detail::Primitive&,
				std::span<const detail::TensorView>,
				detail::MutableTensorView)
			{
				++calls;
			});

		EXPECT_TRUE(registry.contains(key));
		const detail::TensorSpec spec{ Shape{}, DType::Float32, Device::cpu() };
		const detail::Materialization storage{ make_test_buffer(sizeof(float)), detail::Layout{} };
		const detail::MutableTensorView output{ spec, storage };
		detail::cpu::CpuRuntime runtime;
		const IdentitySpecPrimitive primitive;
		registry.get(key)(runtime, primitive, std::span<const detail::TensorView>{}, output);
		EXPECT_EQ(calls, 1);
	}

	TEST(KernelRegistryTest, RejectsMissingEmptyAndDuplicateEntries)
	{
		detail::KernelRegistry registry;
		const detail::KernelKey key{ typeid(IdentitySpecPrimitive), DeviceType::Cpu };
		EXPECT_FALSE(registry.contains(key));
		EXPECT_THROW((void)registry.get(key), std::runtime_error);
		EXPECT_THROW(
			registry.register_kernel(key, detail::KernelFn{}),
			std::invalid_argument);

		detail::KernelFn kernel = [](
			detail::DeviceRuntime&,
			const detail::Primitive&,
			std::span<const detail::TensorView>,
			detail::MutableTensorView) {};
		registry.register_kernel(key, kernel);
		EXPECT_THROW(registry.register_kernel(key, kernel), std::logic_error);
	}
}
