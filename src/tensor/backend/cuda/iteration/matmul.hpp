#pragma once

#include <minitensor/types.hpp>

#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cuda
{
// Host-side geometry for a strided, batched matmul with contiguous output. Vectors are interpreted as [1, K] on the
// left or [K, 1] on the right.
class MatmulPlan final
{
  public:
    using size_type = Shape::size_type;
    using offset_type = Layout::offset_type;
    using stride_type = Layout::stride_type;

    struct MatrixStrides final
    {
        stride_type row;
        stride_type column;

        friend bool operator==(const MatrixStrides &, const MatrixStrides &) = default;
    };

    MatmulPlan(const Shape &lhs_shape, const Layout &lhs_layout, const Shape &rhs_shape, const Layout &rhs_layout,
               const Shape &output_shape, const Layout &output_layout);

    [[nodiscard]] size_type batch_count() const noexcept;
    [[nodiscard]] size_type row_count() const noexcept;
    [[nodiscard]] size_type contraction_size() const noexcept;
    [[nodiscard]] size_type column_count() const noexcept;

    [[nodiscard]] const MatrixStrides &lhs_strides() const noexcept;
    [[nodiscard]] const MatrixStrides &rhs_strides() const noexcept;

    [[nodiscard]] const Shape &batch_shape() const noexcept;
    [[nodiscard]] const Layout &lhs_batch_layout() const noexcept;
    [[nodiscard]] const Layout &rhs_batch_layout() const noexcept;
    [[nodiscard]] offset_type output_offset() const noexcept;

  private:
    size_type row_count_;
    size_type contraction_size_;
    size_type column_count_;
    MatrixStrides lhs_strides_;
    MatrixStrides rhs_strides_;
    Shape batch_shape_;
    Layout lhs_batch_layout_;
    Layout rhs_batch_layout_;
    offset_type output_offset_;
};
} // namespace minitensor::detail::cuda
