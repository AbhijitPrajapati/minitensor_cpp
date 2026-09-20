#include "mean.hpp"

#include <stdexcept>
#include <vector>

#include <minitensor/ops.hpp>

namespace minitensor::detail
{
    MeanPrimitive::MeanPrimitive(
        std::span<const Axis> axes,
        Shape::size_type input_rank,
        bool keep_dim)
        : reduction_(axes, input_rank, keep_dim) {}

    std::string_view MeanPrimitive::name() const noexcept
    {
        return "mean";
    }

    TensorSpec MeanPrimitive::infer(std::span<const TensorSpec> inputs) const
    {
        return reduction_.infer(inputs, name());
    }

    const std::vector<Shape::size_type> &MeanPrimitive::axes() const noexcept
    {
        return reduction_.axes();
    }

    bool MeanPrimitive::keep_dim() const noexcept
    {
        return reduction_.keep_dim();
    }

    std::size_t MeanPrimitive::reduction_size(const Shape &input_shape) const
    {
        return reduction_.reduction_size(input_shape);
    }

    std::vector<std::optional<Tensor>> MeanPrimitive::vjp(
        std::span<const Tensor> inputs,
        const Tensor &,
        const Tensor &output_cotangent) const
    {
        if (inputs.size() != 1)
        {
            throw std::logic_error{"mean VJP expects 1 input"};
        }

        const Tensor &input = inputs.front();
        if (reduction_.axes().empty())
        {
            return {output_cotangent};
        }

        const Tensor normalizer = full(
            Shape{},
            static_cast<float>(reduction_.reduction_size(input.shape())),
            TensorOptions{input.dtype(), input.device()});
        return {
            reduction_.broadcast_reduced_tensor(
                output_cotangent / normalizer,
                input.shape())};
    }
}
