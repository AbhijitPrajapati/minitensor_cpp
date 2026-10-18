#include "registrations.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/ops/elementwise.hpp>
#include <minitensor/ops/manipulation.hpp>

#include "tensor/autograd/vjp_registry.hpp"
#include "tensor/primitives/reduction/logsumexp.hpp"
#include "tensor/primitives/reduction/sum.hpp"

namespace minitensor::detail
{
void register_reduction_vjps(VjpRegistry &registry)
{
    registry.register_rule<SumPrimitive>([](const SumPrimitive &primitive, const VjpContext &context) {
        if (context.inputs.size() != 1)
        {
            throw std::logic_error{"sum VJP expects 1 input"};
        }

        const Shape &input_shape = context.inputs.front().shape();
        Tensor expanded = context.output_cotangent;
        if (!primitive.axes().empty() && !primitive.keep_dim())
        {
            std::vector<Extent> expanded_dimensions(input_shape.dimensions().begin(), input_shape.dimensions().end());
            for (const Shape::size_type axis : primitive.axes())
            {
                expanded_dimensions[axis] = Extent{1};
            }
            expanded = reshape(context.output_cotangent, Shape{std::move(expanded_dimensions)});
        }

        return VjpContributions{expanded.shape() == input_shape ? expanded : broadcast_to(expanded, input_shape)};
    });

    registry.register_rule<LogSumExpPrimitive>([](const LogSumExpPrimitive &primitive, const VjpContext &context) {
        if (context.inputs.size() != 1)
        {
            throw std::logic_error{"log sum exp VJP expects 1 input"};
        }

        const Tensor &input = context.inputs.front();
        Tensor expanded_output = context.output;
        Tensor expanded_cotangent = context.output_cotangent;
        if (!primitive.keep_dim())
        {
            std::vector<Extent> expanded_extents(input.shape().dimensions().begin(), input.shape().dimensions().end());
            for (const Shape::size_type axis : primitive.axes())
            {
                expanded_extents[axis] = Extent{1};
            }

            const Shape expanded_shape{std::move(expanded_extents)};
            expanded_output = reshape(expanded_output, expanded_shape);
            expanded_cotangent = reshape(expanded_cotangent, expanded_shape);
        }

        return VjpContributions{expanded_cotangent * exp(input - expanded_output)};
    });
}
} // namespace minitensor::detail
