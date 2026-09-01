#pragma once

#include <typeindex>

#include <minitensor/types.hpp>

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
}