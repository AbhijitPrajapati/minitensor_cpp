#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/random.hpp>
#include <minitensor/types.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace minitensor::test
{
	TEST(RandomKeyTest, DerivationIsDeterministicAndDistinct)
	{
		const RandomKey key{ 67 };
		const auto first_split = split(key);
		const auto second_split = split(key);

		EXPECT_EQ(first_split, second_split);
		EXPECT_NE(first_split[0], first_split[1]);
		EXPECT_EQ(fold_in(key, 3), fold_in(key, 3));
		EXPECT_NE(fold_in(key, 3), fold_in(key, 4));
	}

	TEST(RandomTest, KeysDetermineUniformSamples)
	{
		const auto [first_key, second_key] = split(RandomKey{ 71 });
		const Tensor first = uniform(Shape{ 32 }, -3.0F, 4.0F, first_key);
		const Tensor repeated = uniform(Shape{ 32 }, -3.0F, 4.0F, first_key);
		const Tensor distinct = uniform(Shape{ 32 }, -3.0F, 4.0F, second_key);

		const std::vector<float> first_values = to_vector(first);
		EXPECT_EQ(first_values, to_vector(repeated));
		EXPECT_NE(first_values, to_vector(distinct));
		EXPECT_TRUE(std::ranges::all_of(first_values, [](float value)
			{
				return value >= -3.0F && value < 4.0F;
			}));
	}

	TEST(RandomTest, NormalValuesHaveReasonableFiniteMoments)
	{
		const std::vector<float> values =
			to_vector(normal(Shape{ 2048 }, 1.5F, 0.75F, RandomKey{ 79 }));

		double sum = 0.0;
		double squared_sum = 0.0;
		for (const float value : values)
		{
			ASSERT_TRUE(std::isfinite(value));
			sum += value;
			squared_sum += static_cast<double>(value) * value;
		}
		const double mean = sum / values.size();
		const double variance = squared_sum / values.size() - mean * mean;

		EXPECT_NEAR(mean, 1.5, 0.1);
		EXPECT_NEAR(std::sqrt(variance), 0.75, 0.1);
	}

	TEST(RandomTest, SupportsDegenerateAndEmptyDistributions)
	{
		const RandomKey key{ 81 };
		EXPECT_FLOAT_EQ(item(uniform(Shape{}, 3.0F, 3.0F, key)), 3.0F);
		EXPECT_FLOAT_EQ(item(normal(Shape{}, -2.0F, 0.0F, key)), -2.0F);
		EXPECT_TRUE(to_vector(uniform(Shape{ 0 }, 0.0F, 1.0F, key)).empty());
		EXPECT_TRUE(to_vector(normal(Shape{ 2, 0 }, 0.0F, 1.0F, key)).empty());
	}

	TEST(RandomTest, RejectsInvalidDistributionParameters)
	{
		const RandomKey key{ 83 };
		EXPECT_THROW(
			(void)uniform(Shape{ 1 }, 2.0F, 1.0F, key),
			std::invalid_argument);
		EXPECT_THROW(
			(void)uniform(
			Shape{ 1 }, 0.0F, std::numeric_limits<float>::infinity(), key),
			std::invalid_argument);
		EXPECT_THROW(
			(void)normal(Shape{ 1 }, 0.0F, -1.0F, key),
			std::invalid_argument);
		EXPECT_THROW(
			(void)normal(
			Shape{ 1 }, std::numeric_limits<float>::quiet_NaN(), 1.0F, key),
			std::invalid_argument);
	}
}
