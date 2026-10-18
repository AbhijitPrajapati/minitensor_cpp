#pragma once

#include <minitensor/types.hpp>

namespace minitensor::detail
{
// This structure contains the details of a tensor independent of its materialization.
struct TensorSpec final
{
    Shape shape;
    DType dtype{DType::Float32};
    Device device{Device::cpu()};

    friend bool operator==(const TensorSpec &, const TensorSpec &) = default;
};

} // namespace minitensor::detail