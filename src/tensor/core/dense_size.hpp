#pragma once

#include <cstddef>

namespace minitensor::detail
{
    struct TensorSpec;
    [[nodiscard]] std::size_t dense_size_bytes(const TensorSpec &spec);
}