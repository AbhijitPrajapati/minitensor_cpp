#pragma once

#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
    class ReshapePrimitive final : public Primitive
    {
    public:
        explicit ReshapePrimitive(Shape shape);
        [[nodiscard]] std::string_view name() const noexcept override;
        [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
        [[nodiscard]] std::optional<Layout> try_derive_shared_layout(const TensorSpec &input_spec, const Layout &input_layout, const TensorSpec &output_spec) const override;
        [[nodiscard]] const Shape &shape() const noexcept;
        [[nodiscard]] std::vector<std::optional<Tensor>> vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &output_cotangent) const override;

    private:
        Shape shape_;
    };
}
