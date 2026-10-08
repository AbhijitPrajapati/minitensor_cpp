#include <minitensor/nn/optimizers/sgd.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/nn/gradients.hpp>
#include <minitensor/nn/modules/linear.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/nn/parameter_tree.hpp>
#include <minitensor/ops/elementwise.hpp>
#include <minitensor/ops/reduction.hpp>
#include <minitensor/random.hpp>

namespace minitensor::nn::test
{
	namespace
	{
		class MixedParameterTree final
		{
		public:
			MixedParameterTree(Parameter shared, Parameter frozen)
				: first_{ shared },
				second_{ std::move(shared) },
				frozen_{ std::move(frozen) }
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

			friend class ParameterTreeAccess;
		};
	}

	TEST(SGDTest, ValidatesLearningRate)
	{
		const SGD zero{ 0.0F };
		EXPECT_FLOAT_EQ(zero.learning_rate(), 0.0F);

		EXPECT_THROW((void)SGD{ -0.1F }, std::invalid_argument);
		EXPECT_THROW(
			(void)SGD{ std::numeric_limits<float>::infinity() },
			std::invalid_argument);
		EXPECT_THROW(
			(void)SGD{ std::numeric_limits<float>::quiet_NaN() },
			std::invalid_argument);
	}

	TEST(SGDTest, AppliesGradientsWithoutChangingTheSourceTree)
	{
		const Linear linear{ 2, 1 };
		const Linear::Parameters parameters = linear.initialize(RandomKey{ 61 });
		ASSERT_NE(parameters.bias(), nullptr);
		const std::vector<float> source_weight = to_vector(parameters.weight().value());
		const std::vector<float> source_bias = to_vector(parameters.bias()->value());
		const Tensor loss = sum(parameters.weight().value() * parameters.weight().value())
			+ sum(parameters.bias()->value() * parameters.bias()->value());
		const ParameterGradients gradients = parameter_gradients(loss, parameters);
		const SGD optimizer{ 0.25F };

		auto [updated, state] = optimizer.step(
			parameters,
			optimizer.initialize(parameters),
			gradients);
		(void)state;

		EXPECT_EQ(updated.weight().id(), parameters.weight().id());
		ASSERT_NE(updated.bias(), nullptr);
		EXPECT_EQ(updated.bias()->id(), parameters.bias()->id());
		const std::vector<float> updated_weight = to_vector(updated.weight().value());
		const std::vector<float> updated_bias = to_vector(updated.bias()->value());
		ASSERT_EQ(updated_weight.size(), source_weight.size());
		ASSERT_EQ(updated_bias.size(), source_bias.size());
		for (std::size_t index = 0; index < source_weight.size(); ++index)
		{
			EXPECT_FLOAT_EQ(updated_weight[index], 0.5F * source_weight[index]);
		}
		for (std::size_t index = 0; index < source_bias.size(); ++index)
		{
			EXPECT_FLOAT_EQ(updated_bias[index], 0.5F * source_bias[index]);
		}
		EXPECT_EQ(to_vector(parameters.weight().value()), source_weight);
		EXPECT_EQ(to_vector(parameters.bias()->value()), source_bias);
	}

	TEST(SGDTest, UpdatesTiedParametersOnceAndLeavesFrozenParametersUnchanged)
	{
		const Parameter shared{ from_data({ 1.0F, 2.0F }, Shape{ 2 }) };
		const Parameter frozen{
			from_data({ 3.0F, 4.0F }, Shape{ 2 }),
			ParameterMetadata{ .trainable = false }
		};
		const MixedParameterTree parameters{ shared, frozen };
		ParameterGradients gradients;
		gradients.emplace(
			shared.id(),
			from_data({ 0.5F, 1.0F }, Shape{ 2 }));
		const SGD optimizer{ 0.5F };

		auto [updated, state] = optimizer.step(
			parameters,
			optimizer.initialize(parameters),
			gradients);
		(void)state;

		EXPECT_EQ(updated.first().id(), shared.id());
		EXPECT_EQ(updated.second().id(), shared.id());
		EXPECT_EQ(
			to_vector(updated.first().value()),
			(std::vector<float>{ 0.75F, 1.5F }));
		EXPECT_EQ(
			to_vector(updated.second().value()),
			(std::vector<float>{ 0.75F, 1.5F }));
		EXPECT_EQ(updated.frozen().id(), frozen.id());
		EXPECT_FALSE(updated.frozen().metadata().trainable);
		EXPECT_EQ(
			to_vector(updated.frozen().value()),
			(std::vector<float>{ 3.0F, 4.0F }));
	}

	TEST(SGDTest, RejectsIncompleteMismatchedAndExtraGradientMaps)
	{
		const Linear linear{ 2, 1 };
		const Linear::Parameters parameters = linear.initialize(RandomKey{ 67 });
		ASSERT_NE(parameters.bias(), nullptr);
		const Tensor loss = sum(parameters.weight().value() * parameters.weight().value())
			+ sum(parameters.bias()->value() * parameters.bias()->value());
		const ParameterGradients valid = parameter_gradients(loss, parameters);
		const SGD optimizer{ 0.1F };
		const SGD::State state = optimizer.initialize(parameters);

		ParameterGradients missing = valid;
		missing.erase(parameters.bias()->id());
		EXPECT_THROW(
			(void)optimizer.step(parameters, state, missing),
			std::invalid_argument);

		ParameterGradients mismatched = valid;
		mismatched.insert_or_assign(
			parameters.weight().id(),
			from_data({ 1.0F }, Shape{}));
		EXPECT_THROW(
			(void)optimizer.step(parameters, state, mismatched),
			std::invalid_argument);

		ParameterGradients extra = valid;
		const Parameter unrelated{ from_data({ 1.0F }, Shape{}) };
		extra.emplace(unrelated.id(), unrelated.value());
		EXPECT_THROW(
			(void)optimizer.step(parameters, state, extra),
			std::invalid_argument);
	}
}
