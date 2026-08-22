#include "minitensor/tensor.hpp"

#include "autograd/node.hpp"
#include "core/layout.hpp"
#include "core/tensor_impl.hpp"
#include "ops/operation_utils.hpp"

#include <optional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace minitensor {
namespace {

std::size_t as_size(const Index value) {
    return static_cast<std::size_t>(value);
}

Index checked_stride_step(const Index stride, const Index step) {
    if (stride != 0 && step > std::numeric_limits<Index>::max() / stride) {
        throw std::overflow_error("slice stride overflows int64_t");
    }
    return stride * step;
}

} // namespace

Tensor Tensor::transpose(const Index dim0, const Index dim1) const {
    if (dim0 < 0 || dim1 < 0 || dim0 >= rank() || dim1 >= rank()) {
        throw std::out_of_range("transpose dimension is out of range");
    }
    if (dim0 == dim1) {
        return *this;
    }

    auto transposed_shape = shape();
    auto transposed_strides = strides();
    std::swap(transposed_shape[as_size(dim0)], transposed_shape[as_size(dim1)]);
    std::swap(transposed_strides[as_size(dim0)], transposed_strides[as_size(dim1)]);

    auto result = Tensor(std::make_shared<detail::TensorImpl>(
        impl_->storage,
        detail::Layout(
            std::move(transposed_shape),
            std::move(transposed_strides),
            storage_offset()),
        requires_grad()));

    if (requires_grad()) {
        detail::set_history(
            result,
            "transpose",
            {*this},
            [dim0, dim1](const Tensor& gradient) {
                return detail::GradList{
                    std::optional<Tensor>{gradient.transpose(dim0, dim1)}};
            });
    }
    return result;
}

Tensor Tensor::slice(
    const Index dim,
    const Index start,
    const Index stop,
    const Index step) const {
    if (dim < 0 || dim >= rank()) {
        throw std::out_of_range("slice dimension is out of range");
    }
    if (step <= 0) {
        throw std::invalid_argument("slice step must be positive");
    }
    const auto dimension = as_size(dim);
    if (start < 0 || stop < 0 || start > stop || stop > shape()[dimension]) {
        throw std::out_of_range("slice bounds are outside the selected dimension");
    }

    auto sliced_shape = shape();
    auto sliced_strides = strides();
    const auto span = stop - start;
    sliced_shape[dimension] = span == 0 ? 0 : 1 + (span - 1) / step;
    sliced_strides[dimension] = checked_stride_step(sliced_strides[dimension], step);

    auto sliced_offset = storage_offset();
    if (span != 0) {
        Coordinates first_coordinates(as_size(rank()), 0);
        first_coordinates[dimension] = start;
        sliced_offset = impl_->layout.offset_from_coordinates(first_coordinates);
    }

    const auto input_shape = shape();
    auto result = Tensor(std::make_shared<detail::TensorImpl>(
        impl_->storage,
        detail::Layout(
            std::move(sliced_shape),
            std::move(sliced_strides),
            sliced_offset),
        requires_grad()));

    if (requires_grad()) {
        detail::set_history(
            result,
            "slice",
            {*this},
            [input_shape, dim, start, step](const Tensor& gradient) {
                return detail::GradList{std::optional<Tensor>{
                    detail::slice_gradient(gradient, input_shape, dim, start, step)}};
            });
    }
    return result;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }

    auto result = detail::make_contiguous_tensor(shape(), requires_grad());
    detail::copy(detail::read_arg(*this), detail::write_arg(result));
    if (requires_grad()) {
        detail::set_history(
            result,
            "contiguous",
            {*this},
            [](const Tensor& gradient) {
                return detail::GradList{std::optional<Tensor>{gradient.detach()}};
            });
    }
    return result;
}

Tensor Tensor::reshape(Shape new_shape) const {
    if (detail::checked_numel(new_shape) != numel()) {
        throw std::invalid_argument("reshape must preserve the number of elements");
    }
    if (!is_contiguous()) {
        return contiguous().reshape(std::move(new_shape));
    }
    if (new_shape == shape()) {
        return *this;
    }

    const auto input_shape = shape();
    auto result = Tensor(std::make_shared<detail::TensorImpl>(
        impl_->storage,
        detail::Layout::contiguous(std::move(new_shape), storage_offset()),
        requires_grad()));
    if (requires_grad()) {
        detail::set_history(
            result,
            "reshape",
            {*this},
            [input_shape](const Tensor& gradient) {
                return detail::GradList{
                    std::optional<Tensor>{gradient.reshape(input_shape)}};
            });
    }
    return result;
}

} // namespace minitensor
