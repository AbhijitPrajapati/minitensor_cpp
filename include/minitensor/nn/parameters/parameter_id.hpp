#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace minitensor::nn
{
// IDs identify parameters independently of their values and metadata.
// Only a builder can allocate a new identity.
class ParameterId final
{
  public:
    using value_type = std::uint64_t;

    ParameterId() = delete;

    [[nodiscard]] constexpr value_type value() const noexcept
    {
        return value_;
    }

    friend constexpr bool operator==(const ParameterId &, const ParameterId &) = default;

  private:
    explicit constexpr ParameterId(value_type value) noexcept : value_{value}
    {
    }

    value_type value_;

    friend class ParameterBuilder;
};
} // namespace minitensor::nn

template <>
struct std::hash<minitensor::nn::ParameterId>
{
    [[nodiscard]] std::size_t operator()(minitensor::nn::ParameterId id) const noexcept
    {
        return std::hash<minitensor::nn::ParameterId::value_type>{}(id.value());
    }
};
