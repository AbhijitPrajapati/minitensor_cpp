#include "autograd/tensor_access.hpp"

#include "autograd/node.hpp"
#include "core/tensor_impl.hpp"

#include <functional>
#include <utility>

namespace minitensor::detail::autograd
{

    TensorId::TensorId(const TensorImpl *const value) noexcept : value_(value) {}

    std::size_t TensorIdHash::operator()(const TensorId &identity) const noexcept
    {
        return std::hash<const TensorImpl *>{}(identity.value_);
    }

    TensorId TensorAutogradAccess::identity(const Tensor &tensor) noexcept
    {
        return TensorId{tensor.impl_.get()};
    }

    const Node *TensorAutogradAccess::grad_fn(const Tensor &tensor) noexcept
    {
        return tensor.impl_->autograd.grad_fn.get();
    }

    void TensorAutogradAccess::set_grad_fn(
        Tensor &tensor,
        std::unique_ptr<Node> node) noexcept
    {
        tensor.impl_->autograd.grad_fn = std::move(node);
    }

    void TensorAutogradAccess::set_grad(
        Tensor &tensor,
        const Tensor &gradient) noexcept
    {
        tensor.impl_->autograd.grad = gradient.impl_;
    }

} // namespace minitensor::detail::autograd
