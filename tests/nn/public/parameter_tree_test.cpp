#include <minitensor/nn/parameter_tree.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/nn/module.hpp>
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
        : first_{std::move(first)}, second_{std::move(second)}
    {
    }

    [[nodiscard]] const Linear::Parameters &first() const noexcept
    {
        return first_;
    }

    [[nodiscard]] const Linear::Parameters &second() const noexcept
    {
        return second_;
    }

  private:
    template <typename Self, typename Visitor>
    static void visit_members(Self &self, Visitor &visitor)
    {
        visitor.child("first", self.first_);
        visitor.child("second", self.second_);
    }

    friend class ::minitensor::nn::ParameterTreeAccess;

    Linear::Parameters first_;
    Linear::Parameters second_;
};

class TiedParameterTree final
{
  public:
    explicit TiedParameterTree(ParameterId parameter) : first_{parameter}, second_{std::move(parameter)}
    {
    }

    [[nodiscard]] const ParameterId &first() const noexcept
    {
        return first_;
    }

    [[nodiscard]] const ParameterId &second() const noexcept
    {
        return second_;
    }

  private:
    template <typename Self, typename Visitor>
    static void visit_members(Self &self, Visitor &visitor)
    {
        visitor.parameter("first", self.first_);
        visitor.parameter("second", self.second_);
    }

    friend class ::minitensor::nn::ParameterTreeAccess;

    ParameterId first_;
    ParameterId second_;
};

class MalformedParameterTree final
{
  public:
    MalformedParameterTree(ParameterId id, std::string first_name, std::string second_name)
        : first_name_{std::move(first_name)}, second_name_{std::move(second_name)}, parameter_{id}
    {
    }

  private:
    template <typename Self, typename Visitor>
    static void visit_members(Self &self, Visitor &visitor)
    {
        visitor.parameter(self.first_name_, self.parameter_);
        visitor.parameter(self.second_name_, self.parameter_);
    }

    friend class ::minitensor::nn::ParameterTreeAccess;

    std::string first_name_;
    std::string second_name_;
    ParameterId parameter_;
};

TEST(ParameterTreeTest, EnumeratesLeafParameters)
{
    const Linear linear{3, 2};
    ParameterBuilder builder;
    const auto parameters = linear.initialize(builder, RandomKey{19});
    std::vector<ParameterId> ids;

    for_each_parameter(parameters, [&](const ParameterId &parameter) { ids.push_back(parameter); });

    ASSERT_EQ(ids.size(), 2);
    EXPECT_EQ(ids[0], parameters.weight());
    ASSERT_NE(parameters.bias(), nullptr);
    EXPECT_EQ(ids[1], *parameters.bias());
}

TEST(ParameterTreeTest, RecursivelyEnumeratesChildTrees)
{
    const Linear first{3, 4};
    const Linear second{4, 2, false};
    ParameterBuilder builder;
    const LinearParameterPair pair{first.initialize(builder, RandomKey{23}), second.initialize(builder, RandomKey{29})};
    std::vector<ParameterId> ids;

    for_each_parameter(pair, [&](const ParameterId &parameter) { ids.push_back(parameter); });

    ASSERT_EQ(ids.size(), 3);
    EXPECT_EQ(ids[0], pair.first().weight());
    ASSERT_NE(pair.first().bias(), nullptr);
    EXPECT_EQ(ids[1], *pair.first().bias());
    EXPECT_EQ(ids[2], pair.second().weight());
}

TEST(ParameterTreeTest, EnumeratesParametersWithStructuralPaths)
{
    const Linear first{3, 4};
    const Linear second{4, 2, false};
    ParameterBuilder builder;
    const LinearParameterPair pair{first.initialize(builder, RandomKey{41}), second.initialize(builder, RandomKey{43})};
    std::vector<std::string> paths;
    std::vector<ParameterId> ids;

    for_each_named_parameter(pair, [&](std::string_view path, const ParameterId &parameter) {
        paths.emplace_back(path);
        ids.push_back(parameter);
    });

    EXPECT_EQ(paths, (std::vector<std::string>{"first.weight", "first.bias", "second.weight"}));
    ASSERT_EQ(ids.size(), 3);
    EXPECT_EQ(ids[0], pair.first().weight());
    ASSERT_NE(pair.first().bias(), nullptr);
    EXPECT_EQ(ids[1], *pair.first().bias());
    EXPECT_EQ(ids[2], pair.second().weight());
}

TEST(ParameterTreeTest, UniqueNamedTraversalUsesFirstPathForTiedParameters)
{
    ParameterBuilder builder;
    const TiedParameterTree tree{builder.add({full(Shape{1}, 1.0F)})};
    std::vector<std::string> paths;

    for_each_unique_named_parameter(tree,
                                    [&](std::string_view path, const ParameterId &) { paths.emplace_back(path); });

    EXPECT_EQ(paths, (std::vector<std::string>{"first"}));
}

TEST(ParameterTreeTest, RejectsInvalidOrDuplicatePaths)
{
    ParameterBuilder builder;
    const auto id = builder.add({full(Shape{}, 1.0F)});
    const auto ignore = [](std::string_view, const ParameterId &) {};

    EXPECT_THROW(for_each_named_parameter(MalformedParameterTree{id, "", "valid"}, ignore), std::logic_error);
    EXPECT_THROW(for_each_named_parameter(MalformedParameterTree{id, "invalid.name", "valid"}, ignore),
                 std::logic_error);
    EXPECT_THROW(for_each_named_parameter(MalformedParameterTree{id, "duplicate", "duplicate"}, ignore),
                 std::logic_error);
}
} // namespace minitensor::nn::test
