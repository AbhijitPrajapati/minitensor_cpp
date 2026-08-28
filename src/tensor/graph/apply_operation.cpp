#include <vector>
#include <span>
#include <stdexcept>
#include "fwd.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "apply_operation.hpp"
#include "ids.hpp"
#include "value.hpp"
#include "origin.hpp"
#include "primitive.hpp"
#include "node.hpp"

namespace minitensor::detail
{
    ValueRef apply_operation(PrimitiveRef primitive, std::span<const ValueRef> inputs)
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

        

        const NodeRef node = std::make_shared<Node>(next_node_id(), std::move(primitive), std::move(owned_inputs));
        Origin origin{ProducedOrigin{node}};
        const ValueRef output = std::make_shared<Value>(next_value_id(), std::move(output_spec), std::move(origin));
        return output;
    }
}