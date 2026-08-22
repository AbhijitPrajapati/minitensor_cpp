#include "core/tensor_access.hpp"

#include "autograd/node.hpp"
#include "core/tensor_impl.hpp"

#include <utility>

namespace minitensor::detail {

ConstTensorView TensorAccess::view(const Tensor& tensor) noexcept {
    return ConstTensorView{tensor.impl_->storage->read(), tensor.impl_->layout};
}

MutableTensorView TensorAccess::mutable_view(Tensor& tensor) noexcept {
    return MutableTensorView{tensor.impl_->storage->write(), tensor.impl_->layout};
}

TensorAccess::Identity TensorAccess::identity(const Tensor& tensor) noexcept {
    return tensor.impl_.get();
}

const Node* TensorAccess::grad_fn(const Tensor& tensor) noexcept {
    return tensor.impl_->autograd.grad_fn.get();
}

void TensorAccess::set_grad_fn(Tensor& tensor, std::unique_ptr<Node> node) noexcept {
    tensor.impl_->autograd.grad_fn = std::move(node);
}

void TensorAccess::set_grad(Tensor& tensor, const Tensor& gradient) noexcept {
    tensor.impl_->autograd.grad = gradient.impl_;
}

} // namespace minitensor::detail
