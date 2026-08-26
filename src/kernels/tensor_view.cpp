#include "kernels/tensor_view.hpp"

#include "core/tensor_impl.hpp"

namespace minitensor::detail::kernel
{

    ConstTensorView TensorViewAccess::view(const Tensor &tensor) noexcept
    {
        return ConstTensorView{tensor.impl_->storage->read(), tensor.impl_->layout};
    }

    MutableTensorView TensorViewAccess::mutable_view(Tensor &tensor) noexcept
    {
        return MutableTensorView{tensor.impl_->storage->write(), tensor.impl_->layout};
    }

} // namespace minitensor::detail::kernel
