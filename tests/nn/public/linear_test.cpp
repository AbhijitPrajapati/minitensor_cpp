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
		const Linear::Parameters parameters = linear.initialize(RandomKey{ 11 });

		EXPECT_EQ(linear.input_features(), 3);
		EXPECT_EQ(linear.output_features(), 2);
		EXPECT_TRUE(linear.use_bias());
		EXPECT_EQ(parameters.weight().value().shape(), (Shape{ 2, 3 }));
		ASSERT_NE(parameters.bias(), nullptr);
		EXPECT_EQ(parameters.bias()->value().shape(), (Shape{ 2 }));
	}

	TEST(LinearTest, CanOmitBias)
	{
		const Linear linear{ 3, 2, false };
		const Linear::Parameters parameters = linear.initialize(RandomKey{ 13 });

		EXPECT_FALSE(linear.use_bias());
		EXPECT_EQ(parameters.weight().value().shape(), (Shape{ 2, 3 }));
		EXPECT_EQ(parameters.bias(), nullptr);
	}

	TEST(LinearTest, RejectsNonPositiveFeatureCounts)
	{
		EXPECT_THROW((void)Linear(0, 2), std::invalid_argument);
		EXPECT_THROW((void)Linear(3, 0), std::invalid_argument);
		EXPECT_THROW((void)Linear(-1, 2), std::invalid_argument);
		EXPECT_THROW((void)Linear(3, -1), std::invalid_argument);
	}

	TEST(LinearTest, SameKeyReproducesInitialValues)
	{
		const RandomKey key{ 17 };
		const Linear linear{ 3, 2 };
		const Linear::Parameters first = linear.initialize(key);
		const Linear::Parameters second = linear.initialize(key);

		EXPECT_EQ(
			to_vector(first.weight().value()),
			to_vector(second.weight().value()));
		ASSERT_NE(first.bias(), nullptr);
		ASSERT_NE(second.bias(), nullptr);
		EXPECT_EQ(
			to_vector(first.bias()->value()),
			to_vector(second.bias()->value()));
	}

	TEST(LinearTest, AppliesWeightAndBias)
	{
		const Linear linear{ 3, 2 };
		const Linear::Parameters parameters = linear.initialize(RandomKey{ 42 });
		const std::vector<float> input_values{
			1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F
		};
		const Tensor input = from_data(input_values, Shape{ 2, 3 });

		const std::vector<float> weight = to_vector(parameters.weight().value());
		ASSERT_NE(parameters.bias(), nullptr);
		const std::vector<float> bias = to_vector(parameters.bias()->value());
		const std::vector<float> output = to_vector(linear(parameters, input));

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

	TEST(LinearTest, RejectsIncompatibleParameterTrees)
	{
		const Linear linear{ 3, 2 };
		const Linear different_output{ 3, 4 };
		const Linear without_bias{ 3, 2, false };
		const Tensor input = from_data({ 1.0F, 2.0F, 3.0F }, Shape{ 1, 3 });

		EXPECT_THROW(
			(void)linear(different_output.initialize(RandomKey{ 43 }), input),
			std::invalid_argument);
		EXPECT_THROW(
			(void)linear(without_bias.initialize(RandomKey{ 47 }), input),
			std::invalid_argument);
		EXPECT_THROW(
			(void)without_bias(linear.initialize(RandomKey{ 53 }), input),
			std::invalid_argument);
	}
}
