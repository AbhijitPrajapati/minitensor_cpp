#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <minitensor/types.hpp>

namespace minitensor::detail
{

template <typename Range>
concept ExtentRange = std::ranges::random_access_range<const Range> && std::ranges::sized_range<const Range> &&
                      std::same_as<std::remove_cv_t<std::ranges::range_value_t<const Range>>, Extent>;

template <ExtentRange Lhs, ExtentRange Rhs>
[[nodiscard]] std::vector<Extent> broadcast_extents(const Lhs &lhs, const Rhs &rhs)
{
    const std::size_t lhs_rank = static_cast<std::size_t>(std::ranges::size(lhs));
    const std::size_t rhs_rank = static_cast<std::size_t>(std::ranges::size(rhs));
    const auto lhs_begin = std::ranges::begin(lhs);
    const auto rhs_begin = std::ranges::begin(rhs);

    const std::size_t output_rank = std::max(lhs_rank, rhs_rank);
    std::vector<Extent> output_extents(output_rank, Extent{1});

    for (std::size_t from_end = 0; from_end < output_rank; ++from_end)
    {
        const Extent lhs_extent =
            from_end < lhs_rank
                ? lhs_begin[static_cast<std::ranges::range_difference_t<const Lhs>>(lhs_rank - 1 - from_end)]
                : Extent{1};
        const Extent rhs_extent =
            from_end < rhs_rank
                ? rhs_begin[static_cast<std::ranges::range_difference_t<const Rhs>>(rhs_rank - 1 - from_end)]
                : Extent{1};

        Extent output_extent;
        if (lhs_extent == rhs_extent)
        {
            output_extent = lhs_extent;
        }
        else if (lhs_extent == 1)
        {
            output_extent = rhs_extent;
        }
        else if (rhs_extent == 1)
        {
            output_extent = lhs_extent;
        }
        else
        {
            throw std::invalid_argument{"tensor shapes cannot be broadcast"};
        }

        output_extents[output_rank - 1 - from_end] = output_extent;
    }

    return output_extents;
}
} // namespace minitensor::detail
