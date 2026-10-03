#include <minitensor/nn/modules/linear.hpp>

#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/random.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::nn::test
{
	TEST(LinearTest, InitializesExpectedParameterShapes)
	{
		const Linear linear{ 3, 2 };

		EXPECT_EQ(linear.input_features(), 3);
		EXPECT_EQ(linear.output_features(), 2);
		EXPECT_EQ(linear.weight().value().shape(), (Shape{ 2, 3 }));
		ASSERT_NE(linear.bias(), nullptr);
		EXPECT_EQ(linear.bias()->value().shape(), (Shape{ 2 }));
	}

	TEST(LinearTest, CanOmitBias)
	{
		const Linear linear{ 3, 2, false };

		EXPECT_EQ(linear.weight().value().shape(), (Shape{ 2, 3 }));
		EXPECT_EQ(linear.bias(), nullptr);
	}

	TEST(LinearTest, RejectsNonPositiveFeatureCounts)
	{
		EXPECT_THROW((void)Linear(0, 2), std::invalid_argument);
		EXPECT_THROW((void)Linear(3, 0), std::invalid_argument);
		EXPECT_THROW((void)Linear(-1, 2), std::invalid_argument);
		EXPECT_THROW((void)Linear(3, -1), std::invalid_argument);
	}

	TEST(LinearTest, ValidatesBeforeConsumingRandomStreams)
	{
		manual_seed(17);
		const Linear expected{ 3, 2 };
		const std::vector<float> expected_weight =
			to_vector(expected.weight().value());
		ASSERT_NE(expected.bias(), nullptr);
		const std::vector<float> expected_bias =
			to_vector(expected.bias()->value());

		manual_seed(17);
		EXPECT_THROW((void)Linear(0, 2), std::invalid_argument);
		const Linear actual{ 3, 2 };

		EXPECT_EQ(to_vector(actual.weight().value()), expected_weight);
		ASSERT_NE(actual.bias(), nullptr);
		EXPECT_EQ(to_vector(actual.bias()->value()), expected_bias);
	}

	TEST(LinearTest, AppliesWeightAndBias)
	{
		manual_seed(42);
		const Linear linear{ 3, 2 };
		const std::vector<float> input_values{
			1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F
		};
		const Tensor input = from_data(input_values, Shape{ 2, 3 });

		const std::vector<float> weight = to_vector(linear.weight().value());
		ASSERT_NE(linear.bias(), nullptr);
		const std::vector<float> bias = to_vector(linear.bias()->value());
		const std::vector<float> output = to_vector(linear(input));

		ASSERT_EQ(output.size(), 4);
		for (std::size_t row = 0; row < 2; ++row)
		{
			for (std::size_t feature = 0; feature < 2; ++feature)
			{
				float expected = bias[feature];
				for (std::size_t input_feature = 0; input_feature < 3; ++input_feature)
				{
					expected += input_values[row * 3 + input_feature]
						* weight[feature * 3 + input_feature];
				}
				EXPECT_NEAR(output[row * 2 + feature], expected, 1.0e-6F);
			}
		}
	}
}
