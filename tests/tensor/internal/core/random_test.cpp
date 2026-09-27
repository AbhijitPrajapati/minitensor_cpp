#include <minitensor/types.hpp>

#include <gtest/gtest.h>

#include "tensor/core/random.hpp"
#include "tensor/execution/generator_registry.hpp"

namespace minitensor::test
{
	TEST(GeneratorTest, ReservesOrderedStreamsAndResetsOnSeed)
	{
		detail::Generator generator{ 17 };
		EXPECT_EQ(generator.reserve_key(), (detail::RandomKey{ 17, 0 }));
		EXPECT_EQ(generator.reserve_key(), (detail::RandomKey{ 17, 1 }));

		generator.manual_seed(29);
		EXPECT_EQ(generator.reserve_key(), (detail::RandomKey{ 29, 0 }));
	}

	TEST(GeneratorRegistryTest, MaintainsIndependentDeviceStreams)
	{
		detail::GeneratorRegistry registry{ 41 };
		EXPECT_EQ(registry.reserve_key(Device::cpu()), (detail::RandomKey{ 41, 0 }));
		EXPECT_EQ(registry.reserve_key(Device::cpu(1)), (detail::RandomKey{ 41, 0 }));
		EXPECT_EQ(registry.reserve_key(Device::cpu()), (detail::RandomKey{ 41, 1 }));
	}

	TEST(GeneratorRegistryTest, ReseedsExistingAndFutureDevices)
	{
		detail::GeneratorRegistry registry{ 41 };
		(void)registry.reserve_key(Device::cpu());
		(void)registry.reserve_key(Device::cpu(1));

		registry.manual_seed(53);
		EXPECT_EQ(registry.reserve_key(Device::cpu()), (detail::RandomKey{ 53, 0 }));
		EXPECT_EQ(registry.reserve_key(Device::cpu(1)), (detail::RandomKey{ 53, 0 }));
		EXPECT_EQ(registry.reserve_key(Device::cpu(2)), (detail::RandomKey{ 53, 0 }));
	}
}
