#pragma once

#include <optional>
#include <string_view>
#include <span>

#include <minitensor/types.hpp>

#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
    class ReshapePrimitive final : public Primitive
    {
    public:
        ReshapePrimitive(Shape shape);
        [[nodiscard]] std::string_view name() const noexcept override;
        [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
        [[nodiscard]] std::optional<Layout> try_derive_shared_layout(const TensorSpec &input_spec, const Layout &input_layout, const TensorSpec &output_spec) const override;
        [[nodiscard]] const Shape &shape() const noexcept;

    private:
        Shape shape_;
    };
}