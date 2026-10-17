#include "registrations.hpp"

#include <optional>
#include <stdexcept>

#include <minitensor/ops/elementwise.hpp>

#include "tensor/autograd/reduce_to_shape.hpp"
#include "tensor/autograd/vjp_registry.hpp"
#include "tensor/primitives/elementwise/absolute_value.hpp"
#include "tensor/primitives/elementwise/add.hpp"
#include "tensor/primitives/elementwise/divide.hpp"
#include "tensor/primitives/elementwise/exponential.hpp"
#include "tensor/primitives/elementwise/hyperbolic_tangent.hpp"
#include "tensor/primitives/elementwise/logaddexp.hpp"
#include "tensor/primitives/elementwise/logarithm.hpp"
#include "tensor/primitives/elementwise/multiply.hpp"
#include "tensor/primitives/elementwise/negate.hpp"
#include "tensor/primitives/elementwise/sigmoid.hpp"
#include "tensor/primitives/elementwise/sign.hpp"
#include "tensor/primitives/elementwise/square_root.hpp"
#include "tensor/primitives/elementwise/subtract.hpp"

namespace minitensor::detail
{
	void register_elementwise_vjps(VjpRegistry& registry)
	{
		registry.register_rule<AbsoluteValuePrimitive>(
			[](const AbsoluteValuePrimitive&, const VjpContext& context)
			{
				if (context.inputs.size() != 1)
				{
					throw std::logic_error{ "absolute value VJP expects 1 input" };
				}
				return VjpContributions{
					context.output_cotangent * sign(context.inputs.front()) };
			});

		registry.register_rule<AddPrimitive>(
			[](const AddPrimitive&, const VjpContext& context)
			{
				if (context.inputs.size() != 2)
				{
					throw std::logic_error{ "add VJP expects 2 inputs" };
				}
				return VjpContributions{
					reduce_to_shape(context.output_cotangent, context.inputs[0].shape()),
					reduce_to_shape(context.output_cotangent, context.inputs[1].shape()) };
			});

		registry.register_rule<DividePrimitive>(
			[](const DividePrimitive&, const VjpContext& context)
			{
				if (context.inputs.size() != 2)
				{
					throw std::logic_error{ "divide VJP expects 2 inputs" };
				}

				Tensor lhs_cotangent = context.output_cotangent / context.inputs[1];
				Tensor rhs_cotangent =
					-(context.output_cotangent * context.output / context.inputs[1]);
				return VjpContributions{
					reduce_to_shape(lhs_cotangent, context.inputs[0].shape()),
					reduce_to_shape(rhs_cotangent, context.inputs[1].shape()) };
			});

		registry.register_rule<ExponentialPrimitive>(
			[](const ExponentialPrimitive&, const VjpContext& context)
			{
				if (context.inputs.size() != 1)
				{
					throw std::logic_error{ "exp VJP expects 1 input" };
				}
				return VjpContributions{ context.output_cotangent * context.output };
			});

		registry.register_rule<HyperbolicTangentPrimitive>(
			[](const HyperbolicTangentPrimitive&, const VjpContext& context)
			{
				if (context.inputs.size() != 1)
				{
					throw std::logic_error{ "tanh VJP expects 1 input" };
				}
				return VjpContributions{
					context.output_cotangent *
					(1.0F - context.output * context.output) };
			});

		registry.register_rule<LogAddExpPrimitive>(
			[](const LogAddExpPrimitive&, const VjpContext& context)
			{
				if (context.inputs.size() != 2)
				{
					throw std::logic_error{ "log add exp VJP expects 2 inputs" };
				}

				const Tensor lhs_weight = sigmoid(context.inputs[0] - context.inputs[1]);
				const Tensor rhs_weight = sigmoid(context.inputs[1] - context.inputs[0]);
				return VjpContributions{
					reduce_to_shape(
						context.output_cotangent * lhs_weight,
						context.inputs[0].shape()),
					reduce_to_shape(
						context.output_cotangent * rhs_weight,
						context.inputs[1].shape()) };
			});

		registry.register_rule<LogarithmPrimitive>(
			[](const LogarithmPrimitive&, const VjpContext& context)
			{
				if (context.inputs.size() != 1)
				{
					throw std::logic_error{ "log VJP expects 1 input" };
				}
				return VjpContributions{
					context.output_cotangent / context.inputs.front() };
			});

		registry.register_rule<MultiplyPrimitive>(
			[](const MultiplyPrimitive&, const VjpContext& context)
			{
				if (context.inputs.size() != 2)
				{
					throw std::logic_error{ "multiply VJP expects 2 inputs" };
				}

				Tensor lhs_cotangent = context.output_cotangent * context.inputs[1];
				Tensor rhs_cotangent = context.output_cotangent * context.inputs[0];
				return VjpContributions{
					reduce_to_shape(lhs_cotangent, context.inputs[0].shape()),
					reduce_to_shape(rhs_cotangent, context.inputs[1].shape()) };
			});

		registry.register_rule<NegatePrimitive>(
			[](const NegatePrimitive&, const VjpContext& context)
			{
				if (context.inputs.size() != 1)
				{
					throw std::logic_error{ "negate VJP expects 1 input" };
				}
				return VjpContributions{ -context.output_cotangent };
			});

		registry.register_rule<SigmoidPrimitive>(
			[](const SigmoidPrimitive&, const VjpContext& context)
			{
				if (context.inputs.size() != 1)
				{
					throw std::logic_error{ "sigmoid VJP expects 1 input" };
				}
				return VjpContributions{
					context.output_cotangent * context.output * (1.0F - context.output) };
			});

		registry.register_rule<SignPrimitive>(
			[](const SignPrimitive&, const VjpContext& context)
			{
				if (context.inputs.size() != 1)
				{
					throw std::logic_error{ "sign VJP expects 1 input" };
				}
				return VjpContributions{ std::nullopt };
			});

		registry.register_rule<SquareRootPrimitive>(
			[](const SquareRootPrimitive&, const VjpContext& context)
			{
				if (context.inputs.size() != 1)
				{
					throw std::logic_error{ "sqrt VJP expects 1 input" };
				}
				return VjpContributions{
					context.output_cotangent / (2.0F * context.output) };
			});

		registry.register_rule<SubtractPrimitive>(
			[](const SubtractPrimitive&, const VjpContext& context)
			{
				if (context.inputs.size() != 2)
				{
					throw std::logic_error{ "subtract VJP expects 2 inputs" };
				}
				return VjpContributions{
					reduce_to_shape(context.output_cotangent, context.inputs[0].shape()),
					reduce_to_shape(-context.output_cotangent, context.inputs[1].shape()) };
			});
	}
}
