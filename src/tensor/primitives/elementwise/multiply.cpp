#include "multiply.hpp"

#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/core/broadcasting.hpp"
#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
std::string_view MultiplyPrimitive::name() const noexcept
{
    return "multiply";
}

TensorSpec MultiplyPrimitive::infer(std::span<const TensorSpec> inputs) const
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument{"multiply expects 2 input tensors"};
    }

    const TensorSpec &lhs = inputs[0];
    const TensorSpec &rhs = inputs[1];

    if (lhs.dtype != rhs.dtype)
    {
        throw std::invalid_argument{"multiply requires matching dtypes"};
    }
    if (!is_floating_point(lhs.dtype))
    {
        throw std::invalid_argument{"multiply requires floating-point inputs"};
    }
    if (lhs.device != rhs.device)
    {
        throw std::invalid_argument{"multiply requires input tensors on the same device"};
    }

    Shape output_shape{broadcast_extents(lhs.shape.dimensions(), rhs.shape.dimensions())};
    return TensorSpec{std::move(output_shape), lhs.dtype, lhs.device};
}

} // namespace minitensor::detail
