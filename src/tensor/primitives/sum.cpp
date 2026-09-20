#include "sum.hpp"

#include <stdexcept>
#include <vector>

namespace minitensor::detail
{
    SumPrimitive::SumPrimitive(std::span<const Axis> axes, Shape::size_type input_rank, bool keep_dim)
        : reduction_(axes, input_rank, keep_dim) {}

    std::string_view SumPrimitive::name() const noexcept
    {
        return "sum";
    }

    TensorSpec SumPrimitive::infer(std::span<const TensorSpec> inputs) const
    {
        return reduction_.infer(inputs, name());
    }

    const std::vector<Shape::size_type> &SumPrimitive::axes() const noexcept
    {
        return reduction_.axes();
    }

    bool SumPrimitive::keep_dim() const noexcept
    {
        return reduction_.keep_dim();
    }

    std::vector<std::optional<Tensor>> SumPrimitive::vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &output_cotangent) const
    {
        if (inputs.size() != 1)
        {
            throw std::logic_error{"sum VJP expects 1 input"};
        }

        return {
            reduction_.broadcast_reduced_tensor(
                output_cotangent,
                inputs.front().shape())};
    }
}
