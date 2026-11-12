#pragma once

#include <span>

#include <minitensor/types.hpp>

namespace minitensor::detail
{
[[nodiscard]] Shape reduction_shape(const Shape &input_shape, std::span<const Shape::size_type> reduced_axes,
                                    bool keep_dim);
}