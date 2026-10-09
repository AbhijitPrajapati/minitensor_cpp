#include <minitensor/nn/leafification.hpp>

#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/nn/parameter_tree.hpp>
#include <minitensor/ops/creation.hpp>

namespace minitensor::nn::test
{
	namespace
	{
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

			Parameter first_;
			Parameter second_;

			friend class ParameterTreeAccess;
		};
	}

	TEST(ParameterLeafificationTest, PreservesParameterStateAndTies)
	{
		const Parameter shared{
			full(Shape{ 2 }, 4.0F),
			ParameterMetadata{ .trainable = false }
		};
		const TiedParameterTree source{ shared };

		const TiedParameterTree leafified = leafify_parameter_values(source);

		EXPECT_EQ(leafified.first().id(), source.first().id());
		EXPECT_EQ(leafified.second().id(), source.second().id());
		EXPECT_EQ(leafified.first().metadata(), source.first().metadata());
		EXPECT_EQ(leafified.second().metadata(), source.second().metadata());
		EXPECT_EQ(
			to_vector(leafified.first().value()),
			(std::vector<float>{ 4.0F, 4.0F }));
		EXPECT_EQ(
			to_vector(leafified.second().value()),
			(std::vector<float>{ 4.0F, 4.0F }));
		EXPECT_EQ(
			to_vector(source.first().value()),
			(std::vector<float>{ 4.0F, 4.0F }));
	}
}
