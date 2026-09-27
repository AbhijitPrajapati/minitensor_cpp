#include <minitensor/types.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>

#include "tensor/backend/device_runtime.hpp"
#include "tensor/execution/runtime_registry.hpp"
#include "tensor/support/test_runtime.hpp"

namespace minitensor::test
{
	TEST(RuntimeRegistryTest, OwnsAndRetrievesRuntimesByDevice)
	{
		detail::RuntimeRegistry registry;
		auto runtime = std::make_unique<TestRuntime>(Device::cpu(3));
		TestRuntime* address = runtime.get();
		registry.register_runtime(std::move(runtime));

		EXPECT_EQ(runtime, nullptr);
		EXPECT_TRUE(registry.contains(Device::cpu(3)));
		EXPECT_EQ(&registry.get(Device::cpu(3)), address);
		const detail::RuntimeRegistry& const_registry = registry;
		EXPECT_EQ(&const_registry.get(Device::cpu(3)), address);
	}

	TEST(RuntimeRegistryTest, DistinguishesDeviceIndices)
	{
		detail::RuntimeRegistry registry;
		auto first = std::make_unique<TestRuntime>(Device::cpu());
		auto second = std::make_unique<TestRuntime>(Device::cpu(1));
		TestRuntime* first_address = first.get();
		TestRuntime* second_address = second.get();
		registry.register_runtime(std::move(first));
		registry.register_runtime(std::move(second));

		EXPECT_EQ(&registry.get(Device::cpu()), first_address);
		EXPECT_EQ(&registry.get(Device::cpu(1)), second_address);
	}

	TEST(RuntimeRegistryTest, RejectsMissingEmptyAndDuplicateEntries)
	{
		detail::RuntimeRegistry registry;
		EXPECT_THROW((void)registry.get(Device::cpu()), std::runtime_error);
		EXPECT_THROW(
			registry.register_runtime(std::unique_ptr<detail::DeviceRuntime>{}),
			std::invalid_argument);

		registry.register_runtime(std::make_unique<TestRuntime>());
		EXPECT_THROW(
			registry.register_runtime(std::make_unique<TestRuntime>()),
			std::logic_error);
	}

	TEST(RuntimeRegistryTest, DestroysOwnedRuntimes)
	{
		bool destroyed = false;
		{
			detail::RuntimeRegistry registry;
			registry.register_runtime(std::make_unique<TestRuntime>(Device::cpu(), &destroyed));
			EXPECT_FALSE(destroyed);
		}
		EXPECT_TRUE(destroyed);
	}
}
