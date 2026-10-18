#pragma once

#include <cstddef>
#include <cstdint>

namespace minitensor::detail
{

class PrimitiveId final
{
  public:
    using value_type = std::uint32_t;

    PrimitiveId() = delete;

    explicit constexpr PrimitiveId(value_type value) noexcept : value_(value)
    {
    }

    [[nodiscard]] constexpr value_type value() const noexcept
    {
        return value_;
    }

    friend constexpr bool operator==(PrimitiveId, PrimitiveId) noexcept = default;

  private:
    value_type value_;
};

struct PrimitiveIdHash final
{
    [[nodiscard]] constexpr std::size_t operator()(PrimitiveId id) const noexcept
    {
        return static_cast<std::size_t>(id.value());
    }
};
} // namespace minitensor::detail
