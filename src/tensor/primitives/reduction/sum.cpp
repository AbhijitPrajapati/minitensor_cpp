#include "sum.hpp"

#include <stdexcept>
#include <vector>

#include "reduction_common.hpp"

namespace minitensor::detail
{
    SumPrimitive::SumPrimitive(std::span<const Axis> axes, Shape::size_type input_rank, bool keep_dim)
        : axes_(normalize_reduction_axes(axes, input_rank)),
          input_rank_(input_rank),
          keep_dim_(keep_dim) {}

    std::string_view SumPrimitive::name() const noexcept
    {
        return "sum";
    }

    TensorSpec SumPrimitive::infer(std::span<const TensorSpec> inputs) const
    {
        return infer_reduction(inputs, axes_, input_rank_, keep_dim_, name());
    }

    const std::vector<Shape::size_type> &SumPrimitive::axes() const noexcept
    {
        return axes_;
    }

    bool SumPrimitive::keep_dim() const noexcept
    {
        return keep_dim_;
    }

    std::vector<std::optional<Tensor>> SumPrimitive::vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &output_cotangent) const
    {
        if (inputs.size() != 1)
        {
            throw std::logic_error{"sum VJP expects 1 input"};
        }

        return {
            broadcast_reduced_tensor(
                output_cotangent,
                inputs.front().shape(),
                axes_,
                keep_dim_)};
    }
}
