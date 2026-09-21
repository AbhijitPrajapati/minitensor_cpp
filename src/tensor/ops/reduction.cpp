#include <minitensor/ops/reduction.hpp>

#include <memory>
#include <numeric>
#include <vector>

#include "apply_primitive.hpp"
#include "tensor/primitives/reduction/sum.hpp"

namespace minitensor
{
    namespace
    {
        [[nodiscard]] std::vector<Axis> all_axes(const Tensor &input)
        {
            std::vector<Axis> axes(input.rank());
            std::iota(axes.begin(), axes.end(), Axis{0});
            return axes;
        }
    }

    Tensor sum(const Tensor &input, std::span<const Axis> axes, bool keep_dim)
    {
        if (axes.empty())
        {
            return input;
        }
        return detail::apply_primitive(
            std::make_unique<detail::SumPrimitive>(axes, input.rank(), keep_dim), input);
    }

    Tensor sum(const Tensor &input, bool keep_dim)
    {
        return sum(input, all_axes(input), keep_dim);
    }

}
