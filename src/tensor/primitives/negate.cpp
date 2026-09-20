#include "negate.hpp"

#include <stdexcept>
#include <vector>

#include <minitensor/ops.hpp>

#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
    std::string_view NegatePrimitive::name() const noexcept
    {
        return "negate";
    }

    TensorSpec NegatePrimitive::infer(std::span<const TensorSpec> inputs) const
    {
        if (inputs.size() != 1)
        {
            throw std::invalid_argument{"negate expects 1 input tensor"};
        }
        return inputs[0];
    }

    std::vector<std::optional<Tensor>> NegatePrimitive::vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &output_cotangent) const
    {
        if (inputs.size() != 1)
        {
            throw std::logic_error{"negate VJP expects 1 input"};
        }
        return {-output_cotangent};
    }
}
