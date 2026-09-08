#pragma once

#include <concepts>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace minitensor::detail
{
    template <typename Type>
    concept NonBooleanIntegral = std::integral<std::remove_cv_t<Type>> && !std::same_as<std::remove_cv_t<Type>, bool>;

    template <NonBooleanIntegral Integer>
    [[nodiscard]] Integer checked_multiply(Integer lhs, Integer rhs)
    {
        using Limits = std::numeric_limits<Integer>;
        constexpr Integer max = Limits::max();

        if constexpr (std::unsigned_integral<Integer>)
        {
            // basic check for unsigned integers
            if (lhs != 0 && rhs > max / lhs)
            {
                throw std::overflow_error{"integer multiplication overflow"};
            }
        }
        else
        {
            // extensive check for signed integers
            if (lhs == 0 || rhs == 0)
            {
                return 0;
            }

            constexpr Integer min = Limits::min();

            if (lhs > 0)
            {
                if (rhs > 0)
                {
                    if (lhs > max / rhs)
                    {
                        throw std::overflow_error{"integer multiplication overflow"};
                    }
                }
                else
                {
                    if (rhs < min / lhs)
                    {
                        throw std::overflow_error{"integer multiplication overflow"};
                    }
                }
            }
            else
            {
                if (rhs > 0)
                {
                    if (lhs < min / rhs)
                    {
                        throw std::overflow_error{"integer multiplication overflow"};
                    }
                }
                else
                {
                    if (lhs < max / rhs)
                    {
                        throw std::overflow_error{"integer multiplication overflow"};
                    }
                }
            }
        }

        return lhs * rhs;
    }

    template <NonBooleanIntegral Integer>
    [[nodiscard]] Integer checked_add(Integer lhs, Integer rhs)
    {
        using Limits = std::numeric_limits<Integer>;
        constexpr Integer max = Limits::max();

        if constexpr (std::unsigned_integral<Integer>)
        {
            if (lhs > max - rhs)
            {
                throw std::overflow_error{"integer addition overflow"};
            }
        }
        else
        {
            constexpr Integer min = Limits::min();

            if (rhs > 0 && lhs > max - rhs)
            {
                throw std::overflow_error{"integer addition overflow"};
            }

            if (rhs < 0 && lhs < min - rhs)
            {
                throw std::overflow_error{"integer addition overflow"};
            }
        }

        return lhs + rhs;
    }
}