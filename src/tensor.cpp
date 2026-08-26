#include "minitensor/tensor.hpp"

#include "autograd/engine.hpp"
#include "core/layout.hpp"
#include "core/storage.hpp"
#include "core/tensor_impl.hpp"
#include "kernels/kernels.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace minitensor
{

    Tensor::Tensor(Shape shape, const bool requires_grad)
    {
        auto layout = detail::Layout::contiguous(std::move(shape));
        auto storage = std::make_shared<detail::Storage>(layout.numel());
        impl_ = std::make_shared<detail::TensorImpl>(
            std::move(storage), std::move(layout), requires_grad);
    }

    Tensor::Tensor(std::shared_ptr<detail::TensorImpl> impl) noexcept
        : impl_(std::move(impl)) {}

    Tensor::~Tensor() = default;

    Tensor Tensor::zeros(Shape shape, const bool requires_grad)
    {
        return Tensor(std::move(shape), requires_grad);
    }

    Tensor Tensor::ones(Shape shape, const bool requires_grad)
    {
        auto result = Tensor::zeros(std::move(shape), requires_grad);
        detail::kernel::fill(
            detail::kernel::TensorViewAccess::mutable_view(result), 1.0F);
        return result;
    }

    Tensor Tensor::from_data(
        std::vector<float> values,
        Shape shape,
        const bool requires_grad)
    {
        auto layout = detail::Layout::contiguous(std::move(shape));
        auto storage = std::make_shared<detail::Storage>(std::move(values));
        if (storage->size() != layout.numel())
        {
            throw std::invalid_argument("data length does not match tensor shape");
        }
        return Tensor(std::make_shared<detail::TensorImpl>(
            std::move(storage), std::move(layout), requires_grad));
    }

    const Shape &Tensor::shape() const noexcept { return impl_->layout.shape(); }
    const Strides &Tensor::strides() const noexcept { return impl_->layout.strides(); }
    Index Tensor::storage_offset() const noexcept { return impl_->layout.offset(); }
    Index Tensor::rank() const noexcept { return impl_->layout.rank(); }
    Index Tensor::numel() const noexcept { return impl_->layout.numel(); }
    bool Tensor::is_contiguous() const noexcept { return impl_->layout.is_contiguous(); }
    bool Tensor::requires_grad() const noexcept { return impl_->autograd.requires_grad; }

    std::vector<float> Tensor::to_vector() const
    {
        std::vector<float> values(static_cast<std::size_t>(numel()));
        auto output_layout = detail::Layout::contiguous(shape());
        const detail::kernel::MutableTensorView output{values, output_layout};
        detail::kernel::copy(detail::kernel::TensorViewAccess::view(*this), output);
        return values;
    }

    float Tensor::item() const
    {
        if (numel() != 1)
        {
            throw std::logic_error("item() requires a one-element tensor");
        }
        return impl_->storage->read()[static_cast<std::size_t>(impl_->layout.offset())];
    }

    Tensor Tensor::detach() const
    {
        return Tensor(std::make_shared<detail::TensorImpl>(
            impl_->storage, impl_->layout, false));
    }

    std::optional<Tensor> Tensor::grad() const
    {
        if (!impl_->autograd.grad)
        {
            return std::nullopt;
        }
        return Tensor(impl_->autograd.grad);
    }

    void Tensor::backward() const
    {
        if (numel() != 1)
        {
            throw std::logic_error("backward() without a gradient requires a one-element tensor");
        }
        detail::autograd::run_backward(*this, Tensor::ones(shape()));
    }

    void Tensor::backward(const Tensor &gradient) const
    {
        detail::autograd::run_backward(*this, gradient);
    }

    void Tensor::clear_grad()
    {
        impl_->autograd.grad.reset();
    }

} // namespace minitensor
