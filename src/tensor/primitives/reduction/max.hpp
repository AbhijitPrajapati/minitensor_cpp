#pragma once

#include <string_view>
#include <vector>

#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
    class MaxPrimitive final : public Primitive
    {
    public:
        explicit MaxPrimitive(std::span<const Axis> axes, Shape::size_type input_rank, bool keep_dim);
        [[nodiscard]] std::string_view name() const noexcept override;
        [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
        [[nodiscard]] const std::vector<Shape::size_type> &axes() const noexcept;
        [[nodiscard]] bool keep_dim() const noexcept;

    private:
        std::vector<Shape::size_type> axes_;
        Shape::size_type input_rank_;
        bool keep_dim_;
    };
}
