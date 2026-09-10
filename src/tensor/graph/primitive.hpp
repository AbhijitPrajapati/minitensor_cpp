#pragma once

#include <span>
#include <string_view>
#include <optional>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail
{
    class Primitive
    {
    public:
        virtual ~Primitive() = default;
        [[nodiscard]] virtual std::string_view name() const noexcept = 0;
        [[nodiscard]] virtual TensorSpec infer(std::span<const TensorSpec> inputs) const = 0;
        [[nodiscard]] virtual std::optional<Layout> try_derive_shared_layout(const TensorSpec&, const Layout&, const TensorSpec&) const
        {
            return std::nullopt;
        }
        [[nodiscard]] virtual bool requires_kernel_support() const noexcept
        {
            return true;
        }
    };
}
