#include <minitensor/nn/parameter_tree.hpp>

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
	class LinearParameterPair final
	{
	public:
		LinearParameterPair(Linear::Parameters first, Linear::Parameters second)
			: first_{ std::move(first) }, second_{ std::move(second) }
		{}

		[[nodiscard]] const Linear::Parameters& first() const noexcept
		{
			return first_;
		}

		[[nodiscard]] const Linear::Parameters& second() const noexcept
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

		friend class ParameterTreeAccess;

		Linear::Parameters first_;
		Linear::Parameters second_;
	};

	class TiedParameterTree final
	{
	public:
		explicit TiedParameterTree(Parameter parameter)
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

		friend class ParameterTreeAccess;

		Parameter first_;
		Parameter second_;
	};

	TEST(ParameterTreeTest, EnumeratesLeafParameters)
	{
		const Linear linear{ 3, 2 };
		const Linear::Parameters parameters = linear.initialize(RandomKey{ 19 });
		std::vector<ParameterId> ids;

		for_each_parameter(parameters, [&](const Parameter& parameter)
		{
			ids.push_back(parameter.id());
		});

		ASSERT_EQ(ids.size(), 2);
		EXPECT_EQ(ids[0], parameters.weight().id());
		ASSERT_NE(parameters.bias(), nullptr);
		EXPECT_EQ(ids[1], parameters.bias()->id());
	}

	TEST(ParameterTreeTest, RecursivelyEnumeratesChildTrees)
	{
		const Linear first{ 3, 4 };
		const Linear second{ 4, 2, false };
		const LinearParameterPair pair{
			first.initialize(RandomKey{ 23 }),
			second.initialize(RandomKey{ 29 })
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

	TEST(ParameterTreeTest, TransformsChildParametersWithoutChangingTheSource)
	{
		const Linear first{ 3, 4 };
		const Linear second{ 4, 2, false };
		const LinearParameterPair source{
			first.initialize(RandomKey{ 31 }),
			second.initialize(RandomKey{ 37 })
		};
		std::vector<ParameterId> source_ids;
		std::vector<std::vector<float>> source_values;
		for_each_parameter(source, [&](const Parameter& parameter)
		{
			source_ids.push_back(parameter.id());
			source_values.push_back(to_vector(parameter.value()));
		});

		std::size_t invocation_count = 0;
		const LinearParameterPair transformed = transform_parameter_values(
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

	TEST(ParameterTreeTest, TransformsTiedParametersOncePerIdentity)
	{
		const TiedParameterTree source{ Parameter{
			full(Shape{ 2 }, 1.0F),
			ParameterMetadata{.trainable = false }
		} };
		std::size_t unique_count = 0;
		for_each_unique_parameter(source, [&](const Parameter&)
		{
			++unique_count;
		});

		std::size_t invocation_count = 0;
		const TiedParameterTree transformed = transform_parameter_values(
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
