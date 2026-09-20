#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
    class ReductionAttributes final
    {
    public:
        ReductionAttributes(std::span<const Axis> axes, Shape::size_type input_rank, bool keep_dim);

        [[nodiscard]] TensorSpec infer(
            std::span<const TensorSpec> inputs,
            std::string_view operation_name,
            bool accepts_empty_reduction = true) const;
        [[nodiscard]] const std::vector<Shape::size_type> &axes() const noexcept;
        [[nodiscard]] bool keep_dim() const noexcept;
        [[nodiscard]] std::size_t reduction_size(const Shape &input_shape) const;
        [[nodiscard]] Tensor expand_reduced_tensor(
            const Tensor &reduced,
            const Shape &input_shape) const;
        [[nodiscard]] Tensor broadcast_reduced_tensor(
            const Tensor &reduced,
            const Shape &input_shape) const;

    private:
        std::vector<Shape::size_type> axes_;
        Shape::size_type input_rank_;
        bool keep_dim_;
    };
}
