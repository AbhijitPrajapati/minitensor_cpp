#include "apply_primitive.hpp"

#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fwd.hpp"
#include "node.hpp"
#include "primitive.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "value.hpp"

namespace minitensor::detail
{
ValueRef apply_primitive(std::unique_ptr<Primitive> primitive, std::span<const ValueRef> inputs)
{
    if (!primitive)
    {
        throw std::invalid_argument{"apply_primitive requires a primitive"};
    }

    std::vector<TensorSpec> input_specs;
    input_specs.reserve(inputs.size());

    std::vector<ValueRef> owned_inputs;
    owned_inputs.reserve(inputs.size());

    for (const ValueRef &input : inputs)
    {
        if (!input)
        {
            throw std::invalid_argument{"primitive inputs cannot be null"};
        }

        input_specs.push_back(input->spec());
        owned_inputs.push_back(input);
    }

    TensorSpec output_spec = primitive->infer(input_specs);
    const NodeRef node = std::make_shared<Node>(std::move(primitive), std::move(owned_inputs));
    return std::make_shared<Value>(std::move(output_spec), std::move(node));
}
} // namespace minitensor::detail