#pragma once

#include <typeindex>
#include <cstddef>

#include <minitensor/types.hpp>

#include "tensor/core/hash.hpp"

namespace minitensor::detail
{
    using PrimitiveTypeId = std::type_index;

    struct KernelKey final
    {
        PrimitiveTypeId primitive_type;
        DeviceType device_type;
        DType dtype;
        friend bool operator==(const KernelKey &, const KernelKey &) = default;
    };

    struct KernelKeyHash final
    {
        [[nodiscard]] std::size_t operator()(const KernelKey &key) const noexcept
        {
            std::size_t result = key.primitive_type.hash_code();
            combine_hash(result, key.device_type);
            combine_hash(result, key.dtype);
            return result;
        }
    };
}