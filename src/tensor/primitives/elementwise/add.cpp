#include "add.hpp"

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
std::string_view AddPrimitive::name() const noexcept
{
    return "add";
}

TensorSpec AddPrimitive::infer(std::span<const TensorSpec> inputs) const
{
    if (inputs.size() != 2)
    {
        throw std::invalid_argument{"add expects 2 input tensors"};
    }

    const TensorSpec &lhs = inputs[0];
    const TensorSpec &rhs = inputs[1];

    if (lhs.dtype != rhs.dtype)
    {
        throw std::invalid_argument{"add requires matching dtypes"};
    }
    if (!is_floating_point(lhs.dtype))
    {
        throw std::invalid_argument{"add requires floating-point inputs"};
    }
    if (lhs.device != rhs.device)
    {
        throw std::invalid_argument{"add requires input tensors on the same device"};
    }

    Shape output_shape{broadcast_extents(lhs.shape, rhs.shape)};
    return TensorSpec{std::move(output_shape), lhs.dtype, lhs.device};
}

} // namespace minitensor::detail
