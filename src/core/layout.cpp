#include "core/layout.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace minitensor::detail {

namespace {
// Checks for overflow before multiplying
Index checked_multiply(const Index lhs, const Index rhs) {
    if (lhs < 0 || rhs < 0) {
        throw std::invalid_argument("tensor dimensions, strides, and offsets must be non-negative");
    }
    if (lhs != 0 && rhs > std::numeric_limits<Index>::max() / lhs) {
        throw std::overflow_error("tensor size or stride overflows int64_t");
    }
    return lhs * rhs;
}

// Checks for overflow before adding
Index checked_add(const Index lhs, const Index rhs) {
    if (lhs < 0 || rhs < 0) {
        throw std::invalid_argument("tensor dimensions, strides, and offsets must be non-negative");
    }
    if (rhs > std::numeric_limits<Index>::max() - lhs) {
        throw std::overflow_error("tensor storage offset overflows int64_t");
    }
    return lhs + rhs;
}

} // namespace


Index checked_numel(const Shape& shape) {
    return std::accumulate(shape.begin(), shape.end(), Index{1}, checked_multiply);
}

Layout Layout::contiguous(Shape shape, const Index offset) {
    Strides strides(shape.size());
    std::exclusive_scan(
        shape.rbegin(), shape.rend(), strides.rbegin(), Index{1}, checked_multiply);
    return Layout(std::move(shape), std::move(strides), offset);
}

Layout::Layout(Shape shape, Strides strides, const Index offset)
    : shape_(std::move(shape)),
      strides_(std::move(strides)),
      offset_(offset),
      numel_(checked_numel(shape_)) {
    if (shape_.size() != strides_.size()) {
        throw std::invalid_argument("shape and strides must have the same rank");
    }
    if (offset_ < 0 || std::ranges::any_of(strides_, [](const Index stride) { return stride < 0; })) {
        throw std::invalid_argument("layout strides and offset must be non-negative");
    }
}

const Shape& Layout::shape() const noexcept { return shape_; }
const Strides& Layout::strides() const noexcept { return strides_; }
Index Layout::offset() const noexcept { return offset_; }
Index Layout::rank() const noexcept { return static_cast<Index>(shape_.size()); }
Index Layout::numel() const noexcept { return numel_; }

bool Layout::is_contiguous() const noexcept {
    if (numel_ == 0) {
        return true;
    }

    Index expected_stride = 1;
    for (Index i = rank(); i-- > 0;) {
        const auto index = static_cast<std::size_t>(i);
        if (shape_[index] == 1) {
            continue;
        }
        if (strides_[index] != expected_stride) {
            return false;
        }
        expected_stride *= shape_[index];
    }
    return true;
}

Index Layout::maximum_offset() const {
    if (numel_ == 0) {
        return offset_;
    }
    return std::transform_reduce(
        shape_.begin(), 
        shape_.end(), 
        strides_.begin(), 
        offset_, 
        checked_add, 
        [](const Index dim, const Index stride) {
            return checked_multiply(dim - 1, stride);
        });
}

Coordinates Layout::coordinates_from_linear(Index linear) const {
    if (linear < 0 || linear >= numel_) {
        throw std::out_of_range("linear tensor index is outside the layout");
    }

    Coordinates coordinates(shape_.size(), 0);
    for (Index dim = rank(); dim-- > 0;) {
        const auto index = static_cast<std::size_t>(dim);
        coordinates[index] = linear % shape_[index];
        linear /= shape_[index];
    }
    return coordinates;
}

Index Layout::offset_from_coordinates(const std::span<const Index> coordinates) const {
    if (coordinates.size() != shape_.size()) {
        throw std::invalid_argument("coordinate rank does not match layout rank");
    }
    auto result = offset_;
    for (std::size_t dim = 0; dim < coordinates.size(); ++dim) {
        if (coordinates[dim] < 0 || coordinates[dim] >= shape_[dim]) {
            throw std::out_of_range("tensor coordinate is outside the layout");
        }
        result = checked_add(
            result, checked_multiply(coordinates[dim], strides_[dim]));
    }
    return result;
}

} // namespace minitensor::detail
