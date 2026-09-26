#include <minitensor/ops/elementwise.hpp>

#include <memory>

#include <minitensor/ops/creation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "apply_primitive.hpp"
#include "tensor/primitives/elementwise/add.hpp"
#include "tensor/primitives/elementwise/divide.hpp"
#include "tensor/primitives/elementwise/exponential.hpp"
#include "tensor/primitives/elementwise/hyperbolic_tangent.hpp"
#include "tensor/primitives/elementwise/logarithm.hpp"
#include "tensor/primitives/elementwise/multiply.hpp"
#include "tensor/primitives/elementwise/negate.hpp"
#include "tensor/primitives/elementwise/square_root.hpp"
#include "tensor/primitives/elementwise/subtract.hpp"

namespace minitensor
{
	namespace
	{
		Tensor scalar_like(float value, const Tensor& tensor)
		{
			return full(Shape{}, value, TensorOptions{ tensor.dtype(), tensor.device() });
		}
	}

	Tensor operator+(const Tensor& lhs, const Tensor& rhs)
	{
		return detail::apply_primitive(std::make_unique<detail::AddPrimitive>(), lhs, rhs);
	}

	Tensor operator+(const Tensor& tensor, float scalar)
	{
		return tensor + scalar_like(scalar, tensor);
	}

	Tensor operator+(float scalar, const Tensor& tensor)
	{
		return tensor + scalar;
	}

	Tensor operator-(const Tensor& lhs, const Tensor& rhs)
	{
		return detail::apply_primitive(std::make_unique<detail::SubtractPrimitive>(), lhs, rhs);
	}

	Tensor operator-(const Tensor& tensor, float scalar)
	{
		return tensor - scalar_like(scalar, tensor);
	}

	Tensor operator-(float scalar, const Tensor& tensor)
	{
		return scalar_like(scalar, tensor) - tensor;
	}

	Tensor operator*(const Tensor& lhs, const Tensor& rhs)
	{
		return detail::apply_primitive(std::make_unique<detail::MultiplyPrimitive>(), lhs, rhs);
	}

	Tensor operator*(const Tensor& tensor, float scalar)
	{
		return tensor * scalar_like(scalar, tensor);
	}

	Tensor operator*(float scalar, const Tensor& tensor)
	{
		return tensor * scalar;
	}

	Tensor operator/(const Tensor& lhs, const Tensor& rhs)
	{
		return detail::apply_primitive(std::make_unique<detail::DividePrimitive>(), lhs, rhs);
	}

	Tensor operator/(const Tensor& tensor, float scalar)
	{
		return tensor / scalar_like(scalar, tensor);
	}

	Tensor operator/(float scalar, const Tensor& tensor)
	{
		return scalar_like(scalar, tensor) / tensor;
	}

	Tensor operator-(const Tensor& input)
	{
		return detail::apply_primitive(std::make_unique<detail::NegatePrimitive>(), input);
	}

	Tensor exp(const Tensor& input)
	{
		return detail::apply_primitive(std::make_unique<detail::ExponentialPrimitive>(), input);
	}

	Tensor log(const Tensor& input)
	{
		return detail::apply_primitive(std::make_unique<detail::LogarithmPrimitive>(), input);
	}

	Tensor sqrt(const Tensor& input)
	{
		return detail::apply_primitive(std::make_unique<detail::SquareRootPrimitive>(), input);
	}

	Tensor tanh(const Tensor& input)
	{
		return detail::apply_primitive(
			std::make_unique<detail::HyperbolicTangentPrimitive>(), input);
	}
}
