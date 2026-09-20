#pragma once

#include <span>
#include <string_view>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
    class SumPrimitive final : public Primitive
    {
    public:
        explicit SumPrimitive(std::span<const Axis> axes, Shape::size_type input_rank, bool keep_dim);
        [[nodiscard]] std::string_view name() const noexcept override;
        [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
        [[nodiscard]] const std::vector<Shape::size_type> &axes() const noexcept;
        [[nodiscard]] bool keep_dim() const noexcept;
        [[nodiscard]] std::vector<std::optional<Tensor>> vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &output_cotangent) const override;

    private:
        std::vector<Shape::size_type> axes_;
        Shape::size_type input_rank_;
        bool keep_dim_;
    };
}
