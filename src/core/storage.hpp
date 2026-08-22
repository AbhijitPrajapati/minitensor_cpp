#pragma once

#include "minitensor/types.hpp"

#include <span>
#include <vector>

namespace minitensor::detail {

class Storage final {
public:
    // Constructor from size and values
    explicit Storage(Index size);
    explicit Storage(std::vector<float> values);

    [[nodiscard]] Index size() const noexcept;
    [[nodiscard]] std::span<const float> read() const noexcept;
    [[nodiscard]] std::span<float> write() noexcept;

private:
    std::vector<float> values_;
};

} // namespace minitensor::detail
