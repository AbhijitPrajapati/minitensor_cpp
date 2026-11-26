#pragma once

#include <cstddef>
#include <span>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/fwd.hpp"

namespace minitensor::detail
{
// Host spans contain contiguous logical values, with an exact byte count. Export and transfer evaluate and pack
// noncontiguous sources on their device. All transfers complete before returning; imports and cross-device transfers
// create materialized leaves, preserving shape/dtype but not graph connections.

[[nodiscard]] ValueRef make_value_from_host(TensorSpec spec, std::span<const std::byte> source);
void copy_value_to_host(std::span<std::byte> destination, const ValueRef &value);
[[nodiscard]] ValueRef transfer_value(const ValueRef &source, Device destination);
} // namespace minitensor::detail
