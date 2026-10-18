#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include <minitensor/types.hpp>

namespace minitensor
{
// Host-side value representation used to configure operations
class HostScalar final
{
  public:
    // Usually implicitely constructed in operations for brevity
    HostScalar(float value) noexcept;
    HostScalar(std::int64_t value) noexcept;

    // Integral and floating point constructors
    template <std::integral T>
        requires(!std::same_as<std::remove_cv_t<T>, bool> && !std::same_as<std::remove_cv_t<T>, std::int64_t>)
    HostScalar(T value) : HostScalar(integral(value))
    {
    }

    template <std::floating_point T>
        requires(!std::same_as<std::remove_cv_t<T>, float>)
    HostScalar(T value) : HostScalar(floating(value))
    {
    }

    [[nodiscard]] DType dtype() const noexcept;
    [[nodiscard]] HostScalar cast(DType dtype) const;

    template <TensorElement T>
    [[nodiscard]] T get() const
    {
        const T *const value = std::get_if<T>(&value_);
        if (value == nullptr)
        {
            throw std::logic_error{"scalar type does not match requested element type"};
        }
        return *value;
    }

    friend bool operator==(const HostScalar &, const HostScalar &) = default;

  private:
    template <std::integral T>
    [[nodiscard]] static std::int64_t integral(T value)
    {
        if (!std::in_range<std::int64_t>(value))
        {
            throw std::out_of_range{"integral scalar is outside the int64 range"};
        }
        return static_cast<std::int64_t>(value);
    }

    template <std::floating_point T>
    [[nodiscard]] static float floating(T value)
    {
        // Ensure value is inside of float bounds specifically
        const long double widened = static_cast<long double>(value);
        if (std::isfinite(widened) && (widened < static_cast<long double>(std::numeric_limits<float>::lowest()) ||
                                       widened > static_cast<long double>(std::numeric_limits<float>::max())))
        {
            throw std::out_of_range{"floating scalar is outside the float32 range"};
        }
        return static_cast<float>(value);
    }

    // HostScalar wraps std::variant to cleanly represent multiple types
    // at the interface level.
    std::variant<float, std::int64_t> value_;
};
} // namespace minitensor
