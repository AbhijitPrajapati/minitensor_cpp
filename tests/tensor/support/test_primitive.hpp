#pragma once

#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::test
{
    class IdentitySpecPrimitive final : public detail::Primitive
    {
    public:
        [[nodiscard]] std::string_view name() const noexcept override
        {
            return "test_identity";
        }

        [[nodiscard]] detail::TensorSpec infer(std::span<const detail::TensorSpec> inputs) const override
        {
            if (inputs.size() != 1)
            {
                throw std::invalid_argument{"expected one input"};
            }
            return inputs.front();
        }
    };

    class DestructionTrackedPrimitive final : public detail::Primitive
    {
    public:
        explicit DestructionTrackedPrimitive(bool &destroyed) noexcept : destroyed_{destroyed}
        {
            destroyed_ = false;
        }

        ~DestructionTrackedPrimitive() override
        {
            destroyed_ = true;
        }

        [[nodiscard]] std::string_view name() const noexcept override
        {
            return "test_destruction_tracked_identity";
        }

        [[nodiscard]] detail::TensorSpec infer(std::span<const detail::TensorSpec> inputs) const override
        {
            if (inputs.size() != 1)
            {
                throw std::invalid_argument{"expected one input"};
            }
            return inputs.front();
        }

    private:
        bool &destroyed_;
    };

    class IdentityStorageSharingPrimitive final : public detail::Primitive
    {
    public:
        [[nodiscard]] std::string_view name() const noexcept override
        {
            return "test_identity_storage_sharing";
        }

        [[nodiscard]] detail::TensorSpec infer(std::span<const detail::TensorSpec> inputs) const override
        {
            if (inputs.size() != 1)
            {
                throw std::invalid_argument{"expected one input"};
            }

            return inputs.front();
        }

        [[nodiscard]] std::optional<detail::Layout> try_derive_shared_layout(
            const detail::TensorSpec &,
            const detail::Layout &input_layout,
            const detail::TensorSpec &) const override
        {
            return input_layout;
        }

        [[nodiscard]] bool requires_kernel_support() const noexcept override
        {
            return false;
        }
    };
}
