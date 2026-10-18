#include "registrations.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/ops/manipulation.hpp>

#include "tensor/apply_operation.hpp"
#include "tensor/autograd/reduce_to_shape.hpp"
#include "tensor/autograd/vjp_registry.hpp"
#include "tensor/primitives/manipulation/basic_index.hpp"
#include "tensor/primitives/manipulation/broadcast_to.hpp"
#include "tensor/primitives/manipulation/concatenate.hpp"
#include "tensor/primitives/manipulation/contiguous.hpp"
#include "tensor/primitives/manipulation/gather.hpp"
#include "tensor/primitives/manipulation/permute.hpp"
#include "tensor/primitives/manipulation/reshape.hpp"

namespace minitensor::detail
{
void register_manipulation_vjps(VjpRegistry &registry)
{
    registry.register_rule<BasicIndexPrimitive>([](const BasicIndexPrimitive &primitive, const VjpContext &context) {
        if (context.inputs.size() != 1)
        {
            throw std::logic_error{"basic index primitive VJP expects 1 input"};
        }

        return VjpContributions{apply_operation(std::make_unique<BasicIndexScatterPrimitive>(primitive.parameters()),
                                                context.output_cotangent)};
    });

    registry.register_rule<BasicIndexScatterPrimitive>(
        [](const BasicIndexScatterPrimitive &primitive, const VjpContext &context) {
            if (context.inputs.size() != 1)
            {
                throw std::logic_error{"basic index scatter VJP expects 1 input"};
            }

            return VjpContributions{apply_operation(std::make_unique<BasicIndexPrimitive>(primitive.parameters()),
                                                    context.output_cotangent)};
        });

    registry.register_rule<BroadcastToPrimitive>([](const BroadcastToPrimitive &, const VjpContext &context) {
        if (context.inputs.size() != 1)
        {
            throw std::logic_error{"broadcast_to VJP expects 1 input"};
        }
        return VjpContributions{reduce_to_shape(context.output_cotangent, context.inputs.front().shape())};
    });

    registry.register_rule<ConcatenatePrimitive>([](const ConcatenatePrimitive &primitive, const VjpContext &context) {
        if (context.inputs.empty())
        {
            throw std::logic_error{"concatenate VJP expects at least 1 input"};
        }

        VjpContributions contributions;
        contributions.reserve(context.inputs.size());
        Extent start = 0;
        for (const Tensor &input : context.inputs)
        {
            const Extent length = input.shape()[primitive.axis()];
            contributions.emplace_back(
                slice(context.output_cotangent, static_cast<Axis>(primitive.axis()), start, start + length, 1));
            start += length;
        }
        return contributions;
    });

    registry.register_rule<ContiguousPrimitive>([](const ContiguousPrimitive &, const VjpContext &context) {
        if (context.inputs.size() != 1)
        {
            throw std::logic_error{"contiguous VJP expects 1 input"};
        }
        return VjpContributions{context.output_cotangent};
    });

    registry.register_rule<GatherPrimitive>([](const GatherPrimitive &primitive, const VjpContext &context) {
        if (context.inputs.size() != 2)
        {
            throw std::logic_error{"gather VJP expects 2 inputs"};
        }

        return VjpContributions{
            apply_operation(std::make_unique<GatherScatterAddPrimitive>(context.inputs[0].shape(), primitive.axis()),
                            context.output_cotangent, context.inputs[1]),
            std::nullopt};
    });

    registry.register_rule<GatherScatterAddPrimitive>([](const GatherScatterAddPrimitive &primitive,
                                                         const VjpContext &context) {
        if (context.inputs.size() != 2)
        {
            throw std::logic_error{"gather scatter add VJP expects 2 inputs"};
        }

        return VjpContributions{apply_operation(std::make_unique<GatherPrimitive>(static_cast<Axis>(primitive.axis()),
                                                                                  primitive.target_shape().rank()),
                                                context.output_cotangent, context.inputs[1]),
                                std::nullopt};
    });

    registry.register_rule<PermutePrimitive>([](const PermutePrimitive &primitive, const VjpContext &context) {
        if (context.inputs.size() != 1)
        {
            throw std::logic_error{"permute VJP expects 1 input"};
        }

        const auto &permutation = primitive.permutation();
        std::vector<Axis> inverse_permutation(permutation.size());
        for (Shape::size_type output_axis = 0; output_axis < permutation.size(); ++output_axis)
        {
            const Shape::size_type input_axis = permutation[output_axis];
            inverse_permutation[input_axis] = static_cast<Axis>(output_axis);
        }
        return VjpContributions{permute(context.output_cotangent, inverse_permutation)};
    });

    registry.register_rule<ReshapePrimitive>([](const ReshapePrimitive &, const VjpContext &context) {
        if (context.inputs.size() != 1)
        {
            throw std::logic_error{"reshape VJP expects 1 input"};
        }
        return VjpContributions{reshape(context.output_cotangent, context.inputs.front().shape())};
    });
}
} // namespace minitensor::detail
