#include <minitensor/types.hpp>

#include <array>
#include <memory>
#include <variant>
#include <vector>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/ids.hpp"
#include "tensor/graph/node.hpp"
#include "tensor/graph/origin.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/graph/value.hpp"

#include "../support/test.hpp"
#include "../support/test_primitive.hpp"

namespace minitensor::test
{
    void run_ownership_test()
    {
        using detail::LeafOrigin;
        using detail::Node;
        using detail::NodeId;
        using detail::NodeRef;
        using detail::Primitive;
        using detail::ProducedOrigin;
        using detail::TensorSpec;
        using detail::Value;
        using detail::ValueId;
        using detail::ValueRef;

        const TensorSpec spec{Shape{2}, DType::Float32, Device::cpu()};

        {
            const auto primitive = std::make_shared<IdentitySpecPrimitive>();
            NodeRef producer = std::make_shared<Node>(NodeId{200}, primitive, std::vector<ValueRef>{});
            const std::weak_ptr<const Node> weak_producer{producer};

            {
                const auto output = std::make_shared<Value>(ValueId{200}, spec, ProducedOrigin{producer});
                producer.reset();
                expect(!weak_producer.expired(), "a produced value keeps its producer node alive");
            }

            expect(weak_producer.expired(), "a producer is released when its only output owner is destroyed");
        }

        {
            ValueRef input = std::make_shared<Value>(ValueId{201}, spec, LeafOrigin{});
            auto primitive = std::make_shared<IdentitySpecPrimitive>();
            const std::weak_ptr<Value> weak_input{input};
            const std::weak_ptr<const Primitive> weak_primitive{primitive};
            NodeRef producer = std::make_shared<Node>(NodeId{201}, primitive, std::vector<ValueRef>{input});

            input.reset();
            primitive.reset();
            expect(!weak_input.expired(), "a producer node keeps its input values alive");
            expect(!weak_primitive.expired(), "a producer node keeps its primitive alive");

            producer.reset();
            expect(weak_input.expired(), "destroying a producer releases its input values");
            expect(weak_primitive.expired(), "destroying a producer releases its primitive");
        }

        ValueRef leaf = std::make_shared<Value>(ValueId{202}, spec, LeafOrigin{});
        auto first_primitive = std::make_shared<IdentitySpecPrimitive>();
        ValueRef intermediate;
        {
            const std::array<ValueRef, 1> first_inputs{leaf};
            intermediate = detail::apply_operation(first_primitive, first_inputs);
        }

        auto second_primitive = std::make_shared<IdentitySpecPrimitive>();
        ValueRef final_output;
        {
            const std::array<ValueRef, 1> second_inputs{intermediate};
            final_output = detail::apply_operation(second_primitive, second_inputs);
        }

        NodeRef first_node = std::get<ProducedOrigin>(intermediate->origin()).node;
        NodeRef second_node = std::get<ProducedOrigin>(final_output->origin()).node;
        const std::weak_ptr<Value> weak_leaf{leaf};
        const std::weak_ptr<Value> weak_intermediate{intermediate};
        const std::weak_ptr<Value> weak_final{final_output};
        const std::weak_ptr<const Node> weak_first_node{first_node};
        const std::weak_ptr<const Node> weak_second_node{second_node};
        const std::weak_ptr<const Primitive> weak_first_primitive{first_primitive};
        const std::weak_ptr<const Primitive> weak_second_primitive{second_primitive};

        leaf.reset();
        intermediate.reset();
        first_primitive.reset();
        second_primitive.reset();
        first_node.reset();
        second_node.reset();

        expect(!weak_leaf.expired() && !weak_intermediate.expired(),
               "the final output keeps its transitive input graph alive");
        expect(!weak_first_node.expired() && !weak_second_node.expired(),
               "the final output keeps all producer nodes alive");
        expect(!weak_first_primitive.expired() && !weak_second_primitive.expired(),
               "the final output keeps all primitives alive");

        final_output.reset();
        expect(weak_final.expired(), "the final output is destroyed when its handle is reset");
        expect(weak_second_node.expired() && weak_intermediate.expired() && weak_first_node.expired() &&
                   weak_leaf.expired(),
               "destroying the final output cleans up the graph without ownership cycles");
        expect(weak_first_primitive.expired() && weak_second_primitive.expired(),
               "graph cleanup destroys all owned primitives");
    }
}
