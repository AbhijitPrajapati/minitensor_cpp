#pragma once

#include "minitensor/tensor.hpp"

#include <cstddef>
#include <memory>

namespace minitensor::detail
{
    struct TensorImpl;
}

namespace minitensor::detail::autograd
{

    struct Node;
    struct TensorAutogradAccess;
    struct TensorIdHash;

    // Opaque, non-owning identity for a TensorImpl. Autograd keeps Tensor
    // handles alive for as long as their identities are used.
    class TensorId final
    {
    public:
        friend bool operator==(const TensorId &lhs, const TensorId &rhs) noexcept
        {
            return lhs.value_ == rhs.value_;
        }

    private:
        explicit TensorId(const TensorImpl *value) noexcept;

        const TensorImpl *value_;

        friend struct TensorAutogradAccess;
        friend struct TensorIdHash;
    };

    struct TensorIdHash final
    {
        [[nodiscard]] std::size_t operator()(const TensorId &identity) const noexcept;
    };

    struct TensorAutogradAccess final
    {
        [[nodiscard]] static TensorId identity(const Tensor &tensor) noexcept;
        [[nodiscard]] static const Node *grad_fn(const Tensor &tensor) noexcept;
        static void set_grad_fn(Tensor &tensor, std::unique_ptr<Node> node) noexcept;
        static void set_grad(Tensor &tensor, const Tensor &gradient) noexcept;
    };

} // namespace minitensor::detail::autograd
