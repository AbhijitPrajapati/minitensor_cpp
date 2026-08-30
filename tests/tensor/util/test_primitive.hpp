#include <string_view>
#include <stdexcept>
#include <span>

#include "tensor/graph/primitive.hpp"
#include "tensor/core/tensor_spec.hpp"

namespace minitensor::test
{
    class IdentitySpecPrimitive final : public detail::Primitive
    {
    public:
        std::string_view name() const noexcept override
        {
            return "test_identity";
        }

        detail::TensorSpec infer(std::span<const TensorSpec> inputs) const override
        {
            if (inputs.size() != 1)
            {
                throw std::invalid_argument{"expected one input"};
            }
            return inputs[0];
        }
    };
}
