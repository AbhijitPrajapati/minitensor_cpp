#include "logarithm.hpp"

#include <stdexcept>
#include <vector>

#include <minitensor/ops.hpp>

#include "tensor/core/tensor_spec.hpp"

namespace minitensor::detail
{
    std::string_view LogarithmPrimitive::name() const noexcept
    {
        return "log";
    }

    TensorSpec LogarithmPrimitive::infer(std::span<const TensorSpec> inputs) const
    {
        if (inputs.size() != 1)
        {
            throw std::invalid_argument{"log expects 1 input tensor"};
        }
        return inputs.front();
    }

    std::vector<std::optional<Tensor>> LogarithmPrimitive::vjp(
        std::span<const Tensor> inputs,
        const Tensor &,
        const Tensor &output_cotangent) const
    {
        if (inputs.size() != 1)
        {
            throw std::logic_error{"log VJP expects 1 input"};
        }
        return {output_cotangent / inputs.front()};
    }
}
