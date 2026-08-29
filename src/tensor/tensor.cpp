#include <minitensor/tensor.hpp>

#include <stdexcept>

#include "graph/value.hpp"

namespace minitensor
{
    Tensor::Tensor(std::shared_ptr<detail::Value> value) : value_(value)
    {
        if (!value_)
        {
            throw std::invalid_argument{"value cannot be null"};
        }
    }

    const Shape &Tensor::shape() const noexcept
    {
        return value_->spec().shape;
    }

    std::size_t Tensor::rank() const noexcept
    {
        return value_->spec().shape.rank();
    }

    std::size_t Tensor::numel() const noexcept
    {
        return value_->spec().shape.numel();
    }

    DType Tensor::dtype() const noexcept
    {
        return value_->spec().dtype;
    }

    Device Tensor::device() const noexcept
    {
        return value_->spec().device;
    }
}
