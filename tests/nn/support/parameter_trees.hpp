#pragma once

#include <utility>

#include <minitensor/nn/parameter_tree.hpp>

namespace minitensor::nn::test::support
{
class ParameterPair final
{
  public:
    ParameterPair(ParameterId first, ParameterId second) : first_{std::move(first)}, second_{std::move(second)}
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

    ParameterId first_;
    ParameterId second_;

    friend class ::minitensor::nn::ParameterTreeAccess;
};

class MixedParameterTree final
{
  public:
    MixedParameterTree(ParameterId shared, ParameterId frozen)
        : first_{shared}, second_{std::move(shared)}, frozen_{std::move(frozen)}
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

    [[nodiscard]] const ParameterId &frozen() const noexcept
    {
        return frozen_;
    }

  private:
    template <typename Self, typename Visitor>
    static void visit_members(Self &self, Visitor &visitor)
    {
        visitor.parameter("first", self.first_);
        visitor.parameter("second", self.second_);
        visitor.parameter("frozen", self.frozen_);
    }

    ParameterId first_;
    ParameterId second_;
    ParameterId frozen_;

    friend class ::minitensor::nn::ParameterTreeAccess;
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

    ParameterId first_;
    ParameterId second_;

    friend class ::minitensor::nn::ParameterTreeAccess;
};
} // namespace minitensor::nn::test::support
