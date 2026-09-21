#include <minitensor/ops/linalg.hpp>

#include <memory>

#include "apply_primitive.hpp"
#include "tensor/primitives/linalg/matmul.hpp"

namespace minitensor
{
    Tensor matmul(const Tensor &lhs, const Tensor &rhs)
    {
        return detail::apply_primitive(
            std::make_unique<detail::MatmulPrimitive>(), lhs, rhs);
    }
}
