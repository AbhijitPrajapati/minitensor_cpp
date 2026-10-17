#include <minitensor/ops/linalg.hpp>

#include <memory>

#include <minitensor/tensor.hpp>

#include "tensor/apply_operation.hpp"
#include "tensor/primitives/linalg/matmul.hpp"

namespace minitensor
{
	Tensor matmul(const Tensor& lhs, const Tensor& rhs)
	{
		return detail::apply_operation(
			std::make_unique<detail::MatmulPrimitive>(), lhs, rhs);
	}
}
