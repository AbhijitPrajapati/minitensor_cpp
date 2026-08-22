#include "core/tensor_impl.hpp"

#include "autograd/node.hpp"

#include <stdexcept>
#include <utility>

namespace minitensor::detail {

AutogradMeta::AutogradMeta(const bool requires_grad) : requires_grad(requires_grad) {}
AutogradMeta::~AutogradMeta() = default;

TensorImpl::TensorImpl(
    std::shared_ptr<Storage> storage_value,
    Layout layout_value,
    const bool requires_grad)
    : storage(std::move(storage_value)),
      layout(std::move(layout_value)),
      autograd(requires_grad) {
    if (!storage) {
        throw std::invalid_argument("tensor storage cannot be null");
    }
    if (layout.numel() == 0) {
        if (layout.offset() > storage->size()) {
            throw std::out_of_range("empty tensor layout starts beyond storage");
        }
    } else if (layout.maximum_offset() >= storage->size()) {
        throw std::out_of_range("tensor layout reaches beyond storage");
    }
}

TensorImpl::~TensorImpl() = default;

} // namespace minitensor::detail
