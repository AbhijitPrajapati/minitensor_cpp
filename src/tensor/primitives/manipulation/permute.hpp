#pragma once

#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
    class PermutePrimitive final : public Primitive
    {
    public:
        explicit PermutePrimitive(std::span<const Axis> permutation, Shape::size_type input_rank);
        [[nodiscard]] std::string_view name() const noexcept override;
        [[nodiscard]] bool requires_kernel_support() const noexcept override;
        [[nodiscard]] TensorSpec infer(std::span<const TensorSpec> inputs) const override;
        [[nodiscard]] std::optional<Layout> try_derive_shared_layout(const TensorSpec &input_spec, const Layout &input_layout, const TensorSpec &output_spec) const override;
        [[nodiscard]] const std::vector<Shape::size_type> &permutation() const noexcept;
        [[nodiscard]] std::vector<std::optional<Tensor>> vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &output_cotangent) const override;

    private:
        std::vector<Shape::size_type> permutation_;
        Shape::size_type input_rank_;
    };
}
