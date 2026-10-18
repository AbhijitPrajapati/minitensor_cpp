#pragma once

#include <minitensor/types.hpp>

#include "tensor/core/hash.hpp"
#include "tensor/graph/primitive_id.hpp"

namespace minitensor::detail
{
// Each kernel has a unique key.
struct KernelKey final
{
    PrimitiveId primitive_id;
    DeviceType device_type;
    friend bool operator==(const KernelKey &, const KernelKey &) = default;
};

struct KernelKeyHash final
{
    [[nodiscard]] std::size_t operator()(const KernelKey &key) const noexcept
    {
        std::size_t result = PrimitiveIdHash{}(key.primitive_id);
        combine_hash(result, key.device_type);
        return result;
    }
};
} // namespace minitensor::detail