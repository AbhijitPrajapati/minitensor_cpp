#include <minitensor/nn/module.hpp>

#include <cstddef>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/nn/modules/linear.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/random.hpp>

namespace minitensor::nn::test
{
	class LinearPair final
	{
	public:
		LinearPair(Linear first, Linear second)
			: first_{ std::move(first) }, second_{ std::move(second) }
		{}

		[[nodiscard]] const Linear& first() const noexcept
		{
			return first_;
		}

		[[nodiscard]] const Linear& second() const noexcept
		{
			return second_;
		}

	private:
		template <typename Self, typename Visitor>
		static void visit_members(Self& self, Visitor& visitor)
		{
			visitor.child("first", self.first_);
			visitor.child("second", self.second_);
		}

		friend class ModuleAccess;

		Linear first_;
		Linear second_;
	};

	class TiedParameters final
	{
	public:
		explicit TiedParameters(Parameter parameter)
			: first_{ parameter }, second_{ std::move(parameter) }
		{}

		[[nodiscard]] const Parameter& first() const noexcept
		{
			return first_;
		}

		[[nodiscard]] const Parameter& second() const noexcept
		{
			return second_;
		}

	private:
		template <typename Self, typename Visitor>
		static void visit_members(Self& self, Visitor& visitor)
		{
			visitor.parameter("first", self.first_);
			visitor.parameter("second", self.second_);
		}

		friend class ModuleAccess;

		Parameter first_;
		Parameter second_;
	};

	TEST(ModuleTest, EnumeratesLeafParameters)
	{
		const Linear with_bias{ 3, 2, RandomKey{ 19 } };
		std::vector<ParameterId> ids;

		for_each_parameter(with_bias, [&](const Parameter& parameter)
		{
			ids.push_back(parameter.id());
		});

		ASSERT_EQ(ids.size(), 2);
		EXPECT_EQ(ids[0], with_bias.weight().id());
		ASSERT_NE(with_bias.bias(), nullptr);
		EXPECT_EQ(ids[1], with_bias.bias()->id());
	}

	TEST(ModuleTest, RecursivelyEnumeratesChildModules)
	{
		const LinearPair pair{
			Linear{ 3, 4, RandomKey{ 23 } },
			Linear{ 4, 2, RandomKey{ 29 }, false }
		};
		std::vector<ParameterId> ids;

		for_each_parameter(pair, [&](const Parameter& parameter)
		{
			ids.push_back(parameter.id());
		});

		ASSERT_EQ(ids.size(), 3);
		EXPECT_EQ(ids[0], pair.first().weight().id());
		ASSERT_NE(pair.first().bias(), nullptr);
		EXPECT_EQ(ids[1], pair.first().bias()->id());
		EXPECT_EQ(ids[2], pair.second().weight().id());
	}

	TEST(ModuleTest, TransformsChildParametersWithoutChangingTheSource)
	{
		const LinearPair source{
			Linear{ 3, 4, RandomKey{ 31 } },
			Linear{ 4, 2, RandomKey{ 37 }, false }
		};
		std::vector<ParameterId> source_ids;
		std::vector<std::vector<float>> source_values;
		for_each_parameter(source, [&](const Parameter& parameter)
		{
			source_ids.push_back(parameter.id());
			source_values.push_back(to_vector(parameter.value()));
		});

		std::size_t invocation_count = 0;
		const LinearPair transformed = transform_parameter_values(
			source,
			[&](const Parameter& parameter)
			{
				++invocation_count;
				return full_like(parameter.value(), 5.0F);
			});

		std::vector<ParameterId> transformed_ids;
		std::vector<std::vector<float>> transformed_values;
		for_each_parameter(transformed, [&](const Parameter& parameter)
		{
			transformed_ids.push_back(parameter.id());
			transformed_values.push_back(to_vector(parameter.value()));
		});

		std::vector<std::vector<float>> retained_values;
		for_each_parameter(source, [&](const Parameter& parameter)
		{
			retained_values.push_back(to_vector(parameter.value()));
		});

		EXPECT_EQ(invocation_count, source_ids.size());
		EXPECT_EQ(transformed_ids, source_ids);
		EXPECT_EQ(retained_values, source_values);
		ASSERT_EQ(transformed_values.size(), source_values.size());
		for (std::size_t index = 0; index < transformed_values.size(); ++index)
		{
			EXPECT_EQ(
				transformed_values[index],
				std::vector<float>(source_values[index].size(), 5.0F));
		}
	}

	TEST(ModuleTest, TransformsTiedParametersOncePerIdentity)
	{
		const TiedParameters source{ Parameter{
			full(Shape{ 2 }, 1.0F),
			ParameterMetadata{.trainable = false }
		} };
		std::size_t unique_count = 0;
		for_each_unique_parameter(source, [&](const Parameter&)
		{
			++unique_count;
		});

		std::size_t invocation_count = 0;
		const TiedParameters transformed = transform_parameter_values(
			source,
			[&](const Parameter& parameter)
			{
				++invocation_count;
				return full_like(parameter.value(), 7.0F);
			});

		EXPECT_EQ(unique_count, 1);
		EXPECT_EQ(invocation_count, 1);
		EXPECT_EQ(transformed.first().id(), source.first().id());
		EXPECT_EQ(transformed.second().id(), source.first().id());
		EXPECT_FALSE(transformed.first().metadata().trainable);
		EXPECT_FALSE(transformed.second().metadata().trainable);
		EXPECT_EQ(
			to_vector(transformed.first().value()),
			(std::vector<float>{ 7.0F, 7.0F }));
		EXPECT_EQ(
			to_vector(transformed.second().value()),
			(std::vector<float>{ 7.0F, 7.0F }));
	}
}
