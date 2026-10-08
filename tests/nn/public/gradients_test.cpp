#include <cstddef>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/nn/gradients.hpp>
#include <minitensor/nn/module.hpp>
#include <minitensor/nn/modules/linear.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/ops/elementwise.hpp>
#include <minitensor/ops/reduction.hpp>
#include <minitensor/random.hpp>

namespace minitensor::nn
{
	namespace
	{
		class TiedGradientModule final
		{
		public:
			TiedGradientModule(Parameter shared, Parameter frozen)
				: first_{ shared }, second_{ std::move(shared) }, frozen_{ std::move(frozen) }
			{}

			[[nodiscard]] const Parameter& first() const noexcept
			{
				return first_;
			}

			[[nodiscard]] const Parameter& second() const noexcept
			{
				return second_;
			}

			[[nodiscard]] const Parameter& frozen() const noexcept
			{
				return frozen_;
			}

		private:
			template <typename Self, typename Visitor>
			static void visit_members(Self& self, Visitor& visitor)
			{
				visitor.parameter("first", self.first_);
				visitor.parameter("second", self.second_);
				visitor.parameter("frozen", self.frozen_);
			}

			Parameter first_;
			Parameter second_;
			Parameter frozen_;

			friend class ModuleAccess;
		};
	}

	TEST(ParameterGradientsTest, AssociatesGradientsWithParameterIds)
	{
		const Linear linear{ 2, 1, RandomKey{41} };
		ASSERT_NE(linear.bias(), nullptr);

		const Tensor loss = sum(linear.weight().value() * linear.weight().value())
			+ sum(linear.bias()->value() * linear.bias()->value());
		const ParameterGradients gradients = parameter_gradients(loss, linear);

		ASSERT_EQ(gradients.size(), 2);
		ASSERT_TRUE(gradients.contains(linear.weight().id()));
		ASSERT_TRUE(gradients.contains(linear.bias()->id()));

		const std::vector<float> weight = to_vector(linear.weight().value());
		const std::vector<float> weight_gradient =
			to_vector(gradients.at(linear.weight().id()));
		ASSERT_EQ(weight_gradient.size(), weight.size());
		for (std::size_t index = 0; index < weight.size(); ++index)
		{
			EXPECT_FLOAT_EQ(weight_gradient[index], 2.0F * weight[index]);
		}

		const std::vector<float> bias = to_vector(linear.bias()->value());
		const std::vector<float> bias_gradient =
			to_vector(gradients.at(linear.bias()->id()));
		ASSERT_EQ(bias_gradient.size(), bias.size());
		for (std::size_t index = 0; index < bias.size(); ++index)
		{
			EXPECT_FLOAT_EQ(bias_gradient[index], 2.0F * bias[index]);
		}
	}

	TEST(ParameterGradientsTest, FiltersFrozenAndDeduplicatesTiedParameters)
	{
		const Parameter shared{ from_data({1.0F, 2.0F}, Shape{2}) };
		const Parameter frozen{
			from_data({3.0F, 4.0F}, Shape{2}), ParameterMetadata{.trainable = false} };
		const TiedGradientModule module{ shared, frozen };
		const Tensor loss = sum(module.first().value() + module.second().value()
			+ module.frozen().value());

		const ParameterGradients gradients = parameter_gradients(loss, module);

		ASSERT_EQ(gradients.size(), 1);
		ASSERT_TRUE(gradients.contains(shared.id()));
		EXPECT_FALSE(gradients.contains(frozen.id()));
		EXPECT_EQ(to_vector(gradients.at(shared.id())), (std::vector{ 2.0F, 2.0F }));
	}
}
