#pragma once

#include <span>

#include <minitensor/types.hpp>

#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cuda
{
// Host-side partition into output coordinates and coordinates reduced within each output. Only the outer layout
// includes the input's storage offset.
class ReductionPlan final
{
  public:
    using size_type = Shape::size_type;

    ReductionPlan(const Shape &shape, const Layout &layout, std::span<const size_type> axes);

    [[nodiscard]] size_type output_size() const noexcept
    {
        return outer_shape_.numel();
    }
    [[nodiscard]] size_type reduction_size() const noexcept
    {
        return reduction_shape_.numel();
    }
    [[nodiscard]] const Shape &outer_shape() const noexcept
    {
        return outer_shape_;
    }
    [[nodiscard]] const Shape &reduction_shape() const noexcept
    {
        return reduction_shape_;
    }
    [[nodiscard]] const Layout &outer_layout() const noexcept
    {
        return outer_layout_;
    }
    [[nodiscard]] const Layout &reduction_layout() const noexcept
    {
        return reduction_layout_;
    }

  private:
    Shape outer_shape_;
    Shape reduction_shape_;
    Layout outer_layout_;
    Layout reduction_layout_;
};
} // namespace minitensor::detail::cuda
