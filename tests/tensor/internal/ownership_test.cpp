#include <minitensor/types.hpp>

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/ids.hpp"
#include "tensor/graph/node.hpp"
#include "tensor/graph/value.hpp"

#include "../support/test.hpp"
#include "../support/test_primitive.hpp"

namespace minitensor::test
{
    void run_ownership_test()
    {
        using detail::Node;
        using detail::NodeId;
        using detail::NodeRef;
        using detail::Primitive;
        using detail::TensorSpec;
        using detail::Value;
        using detail::ValueId;
        using detail::ValueRef;

        const TensorSpec spec{Shape{2}, DType::Float32, Device::cpu()};

        {
            NodeRef producer = std::make_shared<Node>(NodeId{200}, std::make_unique<IdentitySpecPrimitive>(),
                                                      std::vector<ValueRef>{});
            const std::weak_ptr<const Node> weak_producer{producer};

            {
                const auto output = std::make_shared<Value>(ValueId{200}, spec, producer);
                producer.reset();
                expect(!weak_producer.expired(), "a produced value keeps its producer node alive");
            }

            expect(weak_producer.expired(), "a producer is released when its only output owner is destroyed");
        }

        {
            ValueRef input = std::make_shared<Value>(ValueId{201}, spec);
            bool primitive_destroyed = false;
            auto primitive = std::make_unique<DestructionTrackedPrimitive>(primitive_destroyed);
            const Primitive *primitive_address = primitive.get();
            const std::weak_ptr<Value> weak_input{input};
            NodeRef producer = std::make_shared<Node>(NodeId{201}, std::move(primitive),
                                                      std::vector<ValueRef>{input});

            input.reset();
            expect(primitive == nullptr, "a producer node takes unique ownership of its primitive");
            expect(!weak_input.expired(), "a producer node keeps its input values alive");
            expect(!primitive_destroyed && &producer->primitive() == primitive_address,
                   "a producer node keeps its primitive alive");

            producer.reset();
            expect(weak_input.expired(), "destroying a producer releases its input values");
            expect(primitive_destroyed, "destroying a producer destroys its owned primitive");
        }

        ValueRef leaf = std::make_shared<Value>(ValueId{202}, spec);
        bool first_primitive_destroyed = false;
        bool second_primitive_destroyed = false;
        auto first_primitive = std::make_unique<DestructionTrackedPrimitive>(first_primitive_destroyed);
        ValueRef intermediate;
        {
            const std::array<ValueRef, 1> first_inputs{leaf};
            intermediate = detail::apply_operation(std::move(first_primitive), first_inputs);
        }

        auto second_primitive = std::make_unique<DestructionTrackedPrimitive>(second_primitive_destroyed);
        ValueRef final_output;
        {
            const std::array<ValueRef, 1> second_inputs{intermediate};
            final_output = detail::apply_operation(std::move(second_primitive), second_inputs);
        }

        NodeRef first_node = intermediate->producer_ref();
        NodeRef second_node = final_output->producer_ref();
        const std::weak_ptr<Value> weak_leaf{leaf};
        const std::weak_ptr<Value> weak_intermediate{intermediate};
        const std::weak_ptr<Value> weak_final{final_output};
        const std::weak_ptr<const Node> weak_first_node{first_node};
        const std::weak_ptr<const Node> weak_second_node{second_node};

        leaf.reset();
        intermediate.reset();
        first_node.reset();
        second_node.reset();

        expect(!weak_leaf.expired() && !weak_intermediate.expired(),
               "the final output keeps its transitive input graph alive");
        expect(!weak_first_node.expired() && !weak_second_node.expired(),
               "the final output keeps all producer nodes alive");
        expect(!first_primitive_destroyed && !second_primitive_destroyed,
               "the final output keeps all primitives alive");

        final_output.reset();
        expect(weak_final.expired(), "the final output is destroyed when its handle is reset");
        expect(weak_second_node.expired() && weak_intermediate.expired() && weak_first_node.expired() &&
                   weak_leaf.expired(),
               "destroying the final output cleans up the graph without ownership cycles");
        expect(first_primitive_destroyed && second_primitive_destroyed,
               "graph cleanup destroys all owned primitives");
    }
}
