#pragma once

#include <string_view>
#include <vector>

#include "tensor/graph/primitive.hpp"
#include "reduction_attributes.hpp"

namespace minitensor::detail
{
    class MeanPrimitive final : public Primitive
    {
    public:
        explicit MeanPrimitive(std::span<const Axis> axes, Shape::size_type input_rank, bool keep_dim);
        [[nodiscard]] std::string_view name() const noexcept override;
        [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
        [[nodiscard]] const std::vector<Shape::size_type> &axes() const noexcept;
        [[nodiscard]] bool keep_dim() const noexcept;
        [[nodiscard]] std::size_t reduction_size(const Shape &input_shape) const;
        [[nodiscard]] std::vector<std::optional<Tensor>> vjp(
            std::span<const Tensor> inputs,
            const Tensor &output,
            const Tensor &output_cotangent) const override;

    private:
        ReductionAttributes reduction_;
    };
}
