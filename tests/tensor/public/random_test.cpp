#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/random.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace minitensor::test
{
	TEST(RandomTest, ReseedingReproducesConstructionOrderedStreams)
	{
		manual_seed(67);
		const Tensor first_uniform = uniform(Shape{ 16 }, -1.0F, 2.0F);
		const Tensor first_normal = normal(Shape{ 16 }, 1.5F, 0.25F);
		const std::vector<float> normal_evaluated_first = to_vector(first_normal);
		const std::vector<float> uniform_evaluated_second = to_vector(first_uniform);

		manual_seed(67);
		const Tensor repeated_uniform = uniform(Shape{ 16 }, -1.0F, 2.0F);
		const Tensor repeated_normal = normal(Shape{ 16 }, 1.5F, 0.25F);

		EXPECT_EQ(to_vector(repeated_uniform), uniform_evaluated_second);
		EXPECT_EQ(to_vector(repeated_normal), normal_evaluated_first);
	}

	TEST(RandomTest, SuccessiveOperationsUseDistinctStreams)
	{
		manual_seed(71);
		const std::vector<float> first = to_vector(uniform(Shape{ 16 }, 0.0F, 1.0F));
		const std::vector<float> second = to_vector(uniform(Shape{ 16 }, 0.0F, 1.0F));
		EXPECT_NE(first, second);
	}

	TEST(RandomTest, UniformValuesRespectBounds)
	{
		manual_seed(73);
		const std::vector<float> values = to_vector(uniform(Shape{ 128 }, -3.0F, 4.0F));
		EXPECT_TRUE(std::ranges::all_of(values, [](float value)
			{
				return value >= -3.0F && value < 4.0F;
			}));
	}

	TEST(RandomTest, NormalValuesHaveReasonableFiniteMoments)
	{
		manual_seed(79);
		const std::vector<float> values = to_vector(normal(Shape{ 2048 }, 1.5F, 0.75F));

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
		EXPECT_FLOAT_EQ(item(uniform(Shape{}, 3.0F, 3.0F)), 3.0F);
		EXPECT_FLOAT_EQ(item(normal(Shape{}, -2.0F, 0.0F)), -2.0F);
		EXPECT_TRUE(to_vector(uniform(Shape{ 0 }, 0.0F, 1.0F)).empty());
		EXPECT_TRUE(to_vector(normal(Shape{ 2, 0 }, 0.0F, 1.0F)).empty());
	}

	TEST(RandomTest, ValidatesParametersBeforeConsumingAStream)
	{
		manual_seed(83);
		EXPECT_THROW((void)uniform(Shape{ 1 }, 2.0F, 1.0F), std::invalid_argument);
		EXPECT_THROW(
			(void)uniform(
				Shape{ 1 }, 0.0F, std::numeric_limits<float>::infinity()),
			std::invalid_argument);
		EXPECT_THROW((void)normal(Shape{ 1 }, 0.0F, -1.0F), std::invalid_argument);
		EXPECT_THROW(
			(void)normal(
				Shape{ 1 }, std::numeric_limits<float>::quiet_NaN(), 1.0F),
			std::invalid_argument);

		const std::vector<float> after_invalid = to_vector(normal(Shape{ 8 }, 0.0F, 1.0F));
		manual_seed(83);
		EXPECT_EQ(after_invalid, to_vector(normal(Shape{ 8 }, 0.0F, 1.0F)));
	}
}
