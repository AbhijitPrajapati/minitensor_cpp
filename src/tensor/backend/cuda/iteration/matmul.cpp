#include "matmul.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor/core/broadcasting.hpp"

namespace minitensor::detail::cuda
{
namespace
{
MatmulPlan::size_type size_from_extent(Extent extent)
{
    if (!std::in_range<MatmulPlan::size_type>(extent))
    {
        throw std::overflow_error{"matmul extent is not representable as size_t"};
    }
    return static_cast<MatmulPlan::size_type>(extent);
}

Layout aligned_batch_layout(const Shape &shape, const Layout &layout, Shape::size_type batch_rank,
                            const Shape &batch_shape)
{
    std::vector<Layout::stride_type> strides(batch_shape.rank(), 0);
    const auto leading_axes = batch_shape.rank() - batch_rank;
    for (Shape::size_type axis = 0; axis < batch_rank; ++axis)
    {
        if (shape[axis] != 1)
        {
            strides[leading_axes + axis] = layout.stride(axis);
        }
    }
    return Layout{std::move(strides), layout.offset()};
}
} // namespace

MatmulPlan::MatmulPlan(const Shape &lhs_shape, const Layout &lhs_layout, const Shape &rhs_shape,
                       const Layout &rhs_layout, const Shape &output_shape, const Layout &output_layout)
{
    if (lhs_shape.rank() == 0 || rhs_shape.rank() == 0)
    {
        throw std::invalid_argument{"matmul requires inputs with rank 1 or greater"};
    }
    if (lhs_layout.rank() != lhs_shape.rank() || rhs_layout.rank() != rhs_shape.rank() ||
        output_layout.rank() != output_shape.rank())
    {
        throw std::invalid_argument{"matmul shape and layout ranks do not match"};
    }

    const bool lhs_is_vector = lhs_shape.rank() == 1;
    const bool rhs_is_vector = rhs_shape.rank() == 1;
    const auto lhs_batch_rank = lhs_is_vector ? 0 : lhs_shape.rank() - 2;
    const auto rhs_batch_rank = rhs_is_vector ? 0 : rhs_shape.rank() - 2;
    const Extent rows = lhs_is_vector ? Extent{1} : lhs_shape[lhs_shape.rank() - 2];
    const Extent contraction = lhs_shape[lhs_shape.rank() - 1];
    const Extent columns = rhs_is_vector ? Extent{1} : rhs_shape[rhs_shape.rank() - 1];
    if (contraction != rhs_shape[rhs_is_vector ? 0 : rhs_shape.rank() - 2])
    {
        throw std::invalid_argument{"matmul contraction dimensions must match"};
    }

    // Validate output shape.
    batch_shape_ = Shape{
        broadcast_extents(lhs_shape.dimensions().first(lhs_batch_rank), rhs_shape.dimensions().first(rhs_batch_rank))};
    std::vector<Extent> expected_output(batch_shape_.begin(), batch_shape_.end());
    if (!lhs_is_vector)
        expected_output.push_back(rows);
    if (!rhs_is_vector)
        expected_output.push_back(columns);
    if (!std::equal(expected_output.begin(), expected_output.end(), output_shape.begin(), output_shape.end()))
    {
        throw std::invalid_argument{"matmul output shape does not match its operands"};
    }
    if (!output_layout.is_contiguous(output_shape))
    {
        throw std::invalid_argument{"CUDA matmul requires contiguous output"};
    }

    row_count_ = size_from_extent(rows);
    contraction_size_ = size_from_extent(contraction);
    column_count_ = size_from_extent(columns);
    lhs_strides_ = MatrixStrides{lhs_is_vector ? stride_type{0} : lhs_layout.stride(lhs_shape.rank() - 2),
                                 lhs_layout.stride(lhs_shape.rank() - 1)};
    rhs_strides_ = MatrixStrides{rhs_is_vector ? rhs_layout.stride(0) : rhs_layout.stride(rhs_shape.rank() - 2),
                                 rhs_is_vector ? stride_type{0} : rhs_layout.stride(rhs_shape.rank() - 1)};
    lhs_batch_layout_ = aligned_batch_layout(lhs_shape, lhs_layout, lhs_batch_rank, batch_shape_);
    rhs_batch_layout_ = aligned_batch_layout(rhs_shape, rhs_layout, rhs_batch_rank, batch_shape_);
    output_offset_ = output_layout.offset();
}

MatmulPlan::size_type MatmulPlan::batch_count() const noexcept
{
    return batch_shape_.numel();
}

MatmulPlan::size_type MatmulPlan::row_count() const noexcept
{
    return row_count_;
}

MatmulPlan::size_type MatmulPlan::contraction_size() const noexcept
{
    return contraction_size_;
}

MatmulPlan::size_type MatmulPlan::column_count() const noexcept
{
    return column_count_;
}

const MatmulPlan::MatrixStrides &MatmulPlan::lhs_strides() const noexcept
{
    return lhs_strides_;
}

const MatmulPlan::MatrixStrides &MatmulPlan::rhs_strides() const noexcept
{
    return rhs_strides_;
}

const Shape &MatmulPlan::batch_shape() const noexcept
{
    return batch_shape_;
}

const Layout &MatmulPlan::lhs_batch_layout() const noexcept
{
    return lhs_batch_layout_;
}

const Layout &MatmulPlan::rhs_batch_layout() const noexcept
{
    return rhs_batch_layout_;
}

MatmulPlan::offset_type MatmulPlan::output_offset() const noexcept
{
    return output_offset_;
}
} // namespace minitensor::detail::cuda
