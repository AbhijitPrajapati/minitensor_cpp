#pragma once

#include <string_view>

#include "tensor/graph/primitive.hpp"
#include "reduction_attributes.hpp"

namespace minitensor::detail
{
    class MinPrimitive final : public Primitive
    {
    public:
        explicit MinPrimitive(std::span<const Axis> axes, Shape::size_type input_rank, bool keep_dim);
        [[nodiscard]] std::string_view name() const noexcept override;
        [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
        [[nodiscard]] const std::vector<Shape::size_type> &axes() const noexcept;
        [[nodiscard]] bool keep_dim() const noexcept;

    private:
        ReductionAttributes reduction_;
    };
}
