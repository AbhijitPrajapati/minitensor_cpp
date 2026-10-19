#include "broadcast_shape.hpp"
#include <cstdint>
#include <stdexcept>

namespace minitensor::detail
{
    Shape broadcast_shape(const Shape &lhs, const Shape &rhs)
    {
        const std::size_t output_rank = std::max(lhs.rank(), rhs.rank());
        std::vector<Extent> output_dimensions(output_rank, Extent{1});

        for (std::size_t from_end = 0; from_end < output_rank; ++from_end)
        {
            const Extent lhs_extent = from_end < lhs.rank() ? lhs[lhs.rank() - 1 - from_end] : Extent{1};
            const Extent rhs_extent = from_end < rhs.rank() ? rhs[rhs.rank() - 1 - from_end] : Extent{1};

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
            output_dimensions[output_rank - 1 - from_end] = output_extent;
        }
        return Shape(std::move(output_dimensions));
    }
}