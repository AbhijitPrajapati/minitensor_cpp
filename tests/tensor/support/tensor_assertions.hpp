#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <ostream>
#include <ranges>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor
{
inline void PrintTo(const Shape &shape, std::ostream *stream)
{
    *stream << "Shape{";
    bool first = true;
    for (const Extent extent : shape)
    {
        if (!first)
        {
            *stream << ", ";
        }
        *stream << extent;
        first = false;
    }
    *stream << '}';
}

inline void PrintTo(const Device &device, std::ostream *stream)
{
    *stream << "Device{" << static_cast<int>(device.type()) << ", " << device.index() << '}';
}
} // namespace minitensor

namespace minitensor::test
{
namespace assertion_defaults
{
template <std::floating_point T>
inline constexpr T default_absolute_tolerance = static_cast<T>(1.0E-6);

template <std::floating_point T>
inline constexpr T default_relative_tolerance = static_cast<T>(1.0E-5);
} // namespace assertion_defaults

template <std::ranges::contiguous_range Expected>
    requires TensorElement<std::remove_cv_t<std::ranges::range_value_t<Expected>>>
void expect_tensor_eq(const Tensor &tensor, const Expected &expected)
{
    using T = std::remove_cv_t<std::ranges::range_value_t<Expected>>;
    const std::vector<T> expected_values(std::ranges::begin(expected), std::ranges::end(expected));
    EXPECT_EQ(to_vector<T>(tensor), expected_values);
}

template <std::ranges::contiguous_range Expected>
    requires std::floating_point<std::remove_cv_t<std::ranges::range_value_t<Expected>>> &&
             TensorElement<std::remove_cv_t<std::ranges::range_value_t<Expected>>>
void expect_tensor_near(
    const Tensor &tensor, const Expected &expected,
    std::remove_cv_t<std::ranges::range_value_t<Expected>> absolute_tolerance =
        assertion_defaults::default_absolute_tolerance<std::remove_cv_t<std::ranges::range_value_t<Expected>>>,
    std::remove_cv_t<std::ranges::range_value_t<Expected>> relative_tolerance =
        assertion_defaults::default_relative_tolerance<std::remove_cv_t<std::ranges::range_value_t<Expected>>>)
{
    using T = std::remove_cv_t<std::ranges::range_value_t<Expected>>;
    const std::vector<T> actual = to_vector<T>(tensor);
    ASSERT_EQ(actual.size(), std::ranges::size(expected));

    for (std::size_t index = 0; index < actual.size(); ++index)
    {
        const T expected_value = *(std::ranges::begin(expected) + index);
        const T tolerance = absolute_tolerance + relative_tolerance * std::fabs(expected_value);
        EXPECT_NEAR(actual[index], expected_value, tolerance) << "at logical element " << index;
    }
}
} // namespace minitensor::test
