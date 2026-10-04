#include <minitensor/nn/module.hpp>

#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/nn/modules/linear.hpp>
#include <minitensor/nn/parameter.hpp>
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
}
