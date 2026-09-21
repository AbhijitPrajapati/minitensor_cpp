#pragma once

#include <span>
#include <string_view>
#include <vector>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
    [[nodiscard]] std::vector<Shape::size_type> normalize_reduction_axes(
        std::span<const Axis> axes,
        Shape::size_type input_rank);

    [[nodiscard]] TensorSpec infer_reduction(
        std::span<const TensorSpec> inputs,
        std::span<const Shape::size_type> axes,
        Shape::size_type input_rank,
        bool keep_dim,
        std::string_view operation_name,
        bool accepts_empty_reduction = true);

    [[nodiscard]] Tensor broadcast_reduced_tensor(
        const Tensor &reduced,
        const Shape &input_shape,
        std::span<const Shape::size_type> axes,
        bool keep_dim);
}
