#include "core/storage.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace minitensor::detail {
namespace {

std::size_t checked_storage_size(const Index size) {
    if (size < 0) {
        throw std::invalid_argument("storage size must be non-negative");
    }
    if (static_cast<std::uint64_t>(size) > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("storage size does not fit size_t on this platform");
    }
    return static_cast<std::size_t>(size);
}

void validate_vector_size(const std::size_t size) {
    if (size > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
        throw std::length_error("storage size does not fit int64_t");
    }
}

} // namespace

Storage::Storage(const Index size)
    : values_(checked_storage_size(size)) {}

Storage::Storage(std::vector<float> values) : values_(std::move(values)) {
    validate_vector_size(values_.size());
}

Index Storage::size() const noexcept {
    return static_cast<Index>(values_.size());
}

std::span<const float> Storage::read() const noexcept {
    return values_;
}

std::span<float> Storage::write() noexcept {
    return values_;
}

} // namespace minitensor::detail
