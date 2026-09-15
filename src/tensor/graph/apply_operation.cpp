#include "apply_operation.hpp"

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "node.hpp"
#include "primitive.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "value.hpp"

namespace minitensor::detail
{
    ValueRef apply_operation(std::unique_ptr<Primitive> primitive, std::span<const ValueRef> inputs)
    {
        if (!primitive)
        {
            throw std::invalid_argument{"apply_operation requires a primitive"};
        }

        std::vector<TensorSpec> input_specs;
        input_specs.reserve(inputs.size());

        std::vector<ValueRef> owned_inputs;
        owned_inputs.reserve(inputs.size());

        for (const ValueRef &input : inputs)
        {
            if (!input)
            {
                throw std::invalid_argument{"operation inputs cannot be null"};
            }
            input_specs.push_back(input->spec());
            owned_inputs.push_back(input);
        }

        TensorSpec output_spec = primitive->infer(input_specs);

        const NodeRef node = std::make_shared<Node>(std::move(primitive), std::move(owned_inputs));
        const ValueRef output = std::make_shared<Value>(std::move(output_spec), std::move(node));
        return output;
    }
}
