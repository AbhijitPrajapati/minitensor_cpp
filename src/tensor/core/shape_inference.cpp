#include "shape_inference.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/types.hpp>

namespace minitensor::detail
{
    namespace
    {
        std::vector<Extent> broadcast_extents(std::size_t output_rank, const Shape &lhs, const Shape &rhs, std::size_t lhs_effective_rank, std::size_t rhs_effective_rank)
        {
            std::vector<Extent> output_dimensions(output_rank, Extent{1});
            for (std::size_t from_end = 0; from_end < output_rank; ++from_end)
            {
                const Extent lhs_extent = from_end < lhs_effective_rank ? lhs[lhs_effective_rank - 1 - from_end] : Extent{1};
                const Extent rhs_extent = from_end < rhs_effective_rank ? rhs[rhs_effective_rank - 1 - from_end] : Extent{1};

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
            return output_dimensions;
        }
    }

    Shape broadcast_shape(const Shape &lhs, const Shape &rhs)
    {
        const std::size_t output_rank = std::max(lhs.rank(), rhs.rank());
        std::vector<Extent> output_dimensions = broadcast_extents(output_rank, lhs, rhs, lhs.rank(), rhs.rank());
        return Shape(std::move(output_dimensions));
    }

    Shape matmul_output_shape(const Shape &lhs, const Shape &rhs)
    {
        if (lhs.rank() == 0 || rhs.rank() == 0)
        {
            throw std::invalid_argument{"matmul requires inputs with rank 1 or greater"};
        }

        const bool lhs_is_vector = lhs.rank() == 1;
        const bool rhs_is_vector = rhs.rank() == 1;

        const Extent lhs_contraction_extent = lhs[lhs.rank() - 1];
        const Extent rhs_contraction_extent = rhs[rhs_is_vector ? 0 : rhs.rank() - 2];
        if (lhs_contraction_extent != rhs_contraction_extent)
        {
            throw std::invalid_argument{"matmul contraction dimensions must match"};
        }

        const std::size_t lhs_batch_rank = lhs_is_vector ? 0 : lhs.rank() - 2;
        const std::size_t rhs_batch_rank = rhs_is_vector ? 0 : rhs.rank() - 2;
        const std::size_t output_batch_rank = std::max(lhs_batch_rank, rhs_batch_rank);

        // std::vector<Extent> output_dimensions(output_batch_rank, Extent{1});
        // for (std::size_t from_end = 0; from_end < output_batch_rank; ++from_end)
        // {
        //     const Extent lhs_extent = from_end < lhs_batch_rank ? lhs[lhs_batch_rank - 1 - from_end] : Extent{1};
        //     const Extent rhs_extent = from_end < rhs_batch_rank ? rhs[rhs_batch_rank - 1 - from_end] : Extent{1};

        //     Extent output_extent;
        //     if (lhs_extent == rhs_extent)
        //     {
        //         output_extent = lhs_extent;
        //     }
        //     else if (lhs_extent == 1)
        //     {
        //         output_extent = rhs_extent;
        //     }
        //     else if (rhs_extent == 1)
        //     {
        //         output_extent = lhs_extent;
        //     }
        //     else
        //     {
        //         throw std::invalid_argument{"matmul batch dimensions cannot be broadcast"};
        //     }
        //     output_dimensions[output_batch_rank - 1 - from_end] = output_extent;
        // }
        std::vector<Extent> output_dimensions = broadcast_extents(output_batch_rank, lhs, rhs, lhs_batch_rank, rhs_batch_rank);

        if (!lhs_is_vector)
        {
            output_dimensions.push_back(lhs[lhs.rank() - 2]);
        }
        if (!rhs_is_vector)
        {
            output_dimensions.push_back(rhs[rhs.rank() - 1]);
        }

        return Shape{std::move(output_dimensions)};
    }
}
