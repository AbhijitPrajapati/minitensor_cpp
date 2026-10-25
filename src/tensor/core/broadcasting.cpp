#include "broadcasting.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <span>

#include <minitensor/types.hpp>

namespace minitensor::detail
{
std::vector<Extent> broadcast_extents(std::span<const Extent> lhs, std::span<const Extent> rhs)
{
    const std::size_t output_rank = std::max(lhs.size(), rhs.size());
    std::vector<Extent> output_extents(output_rank, Extent{1});

    for (std::size_t from_end = 0; from_end < output_rank; ++from_end)
    {
        const Extent lhs_extent = from_end < lhs.size() ? lhs[lhs.size() - 1 - from_end] : Extent{1};
        const Extent rhs_extent = from_end < rhs.size() ? rhs[rhs.size() - 1 - from_end] : Extent{1};

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