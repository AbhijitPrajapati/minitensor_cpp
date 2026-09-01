#include "tensor_view.hpp"

#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::detail
{
    TensorView::TensorView(const TensorSpec &spec, const Materialization &materialization) : spec_(&spec), materialization_(&materialization)
    {
        materialization_->validate(*spec_);
    }

    const Shape &TensorView::shape() const noexcept
    {
        return spec_->shape;
    }

    DType TensorView::dtype() const noexcept
    {
        return spec_->dtype;
    }

    Device TensorView::device() const noexcept
    {
        return spec_->device;
    }

    const Layout &TensorView::layout() const noexcept
    {
        return materialization_->layout();
    }

    const Buffer &TensorView::buffer() const noexcept
    {
        return *materialization_->buffer_ref();
    }

    MutableTensorView::MutableTensorView(const TensorSpec &spec, const Materialization &materialization) : spec_(&spec), materialization_(&materialization)
    {
        materialization_->validate(*spec_);
    }

    const Shape &MutableTensorView::shape() const noexcept
    {
        return spec_->shape;
    }

    DType MutableTensorView::dtype() const noexcept
    {
        return spec_->dtype;
    }

    Device MutableTensorView::device() const noexcept
    {
        return spec_->device;
    }

    const Layout &MutableTensorView::layout() const noexcept
    {
        return materialization_->layout();
    }

    Buffer &MutableTensorView::buffer() const noexcept
    {
        return *materialization_->buffer_ref();
    }
}