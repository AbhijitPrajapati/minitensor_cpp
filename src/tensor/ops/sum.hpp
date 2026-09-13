#pragma once

#include <string_view>
#include <span>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
    class SumPrimitive final : public Primitive
    {
    public:
        explicit SumPrimitive(std::span<const Axis> axes, Shape::size_type input_rank);
        [[nodiscard]] std::string_view name() const noexcept override;
        [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
        [[nodiscard]] const std::vector<Shape::size_type> &axes() const noexcept;

    private:
        std::vector<Shape::size_type> axes_;
        Shape::size_type input_rank_;
    };
}