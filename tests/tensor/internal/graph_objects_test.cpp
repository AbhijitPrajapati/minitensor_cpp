#include <minitensor/types.hpp>

#include <array>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/ids.hpp"
#include "tensor/graph/node.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/tensor_access.hpp"

#include "../support/test.hpp"
#include "../support/test_primitive.hpp"

namespace minitensor::test
{
    void run_graph_objects_test()
    {
        using detail::Node;
        using detail::NodeId;
        using detail::NodeRef;
        using detail::Primitive;
        using detail::TensorSpec;
        using detail::Value;
        using detail::ValueId;
        using detail::ValueRef;

        const std::array value_ids{detail::next_value_id(), detail::next_value_id(), detail::next_value_id()};
        expect(value_ids[0] != value_ids[1] && value_ids[0] != value_ids[2] && value_ids[1] != value_ids[2],
               "value id generation returns unique ids");

        const std::array node_ids{detail::next_node_id(), detail::next_node_id(), detail::next_node_id()};
        expect(node_ids[0] != node_ids[1] && node_ids[0] != node_ids[2] && node_ids[1] != node_ids[2],
               "node id generation returns unique ids");

        const TensorSpec spec{Shape{2, 3}, DType::Float32, Device::cpu(4)};
        const Value leaf{ValueId{40}, spec};
        expect(leaf.id() == ValueId{40}, "a value preserves its id");
        expect(leaf.spec() == spec, "a value preserves its tensor specification");
        expect(leaf.is_leaf(), "the leaf constructor creates a leaf value");
        expect(leaf.producer() == nullptr, "a leaf value has no producer");
        expect(!leaf.producer_ref(), "a leaf value has no owning producer reference");
        expect(leaf.materialization() == nullptr, "a newly constructed value has no materialization");

        const ValueRef first = std::make_shared<Value>(ValueId{41}, spec);
        const ValueRef second = std::make_shared<Value>(ValueId{42}, spec);
        auto primitive = std::make_unique<IdentitySpecPrimitive>();
        const Primitive *primitive_address = primitive.get();
        const Node node{NodeId{50}, std::move(primitive), {first, second}};

        expect(node.id() == NodeId{50}, "a node preserves its id");
        expect(node.primitive().name() == "test_identity", "a node exposes its primitive");
        expect(primitive == nullptr, "node construction takes ownership of its primitive");
        expect(&node.primitive() == primitive_address, "a node retains the supplied primitive object");
        expect(node.inputs().size() == 2, "a node preserves its input count");
        expect(node.inputs()[0].get() == first.get() && node.inputs()[1].get() == second.get(),
               "a node preserves input order and identity");

        const Tensor handle = detail::TensorAccess::make(first);
        expect(detail::TensorAccess::value(handle).get() == first.get(),
               "a tensor handle retains its internal value");
        expect_throws<std::invalid_argument>(
            []
            {
                (void)detail::TensorAccess::make(ValueRef{});
            },
            "tensor handle construction rejects a null value");

        const auto producer = std::make_shared<Node>(NodeId{51}, std::make_unique<IdentitySpecPrimitive>(),
                                                     std::vector<ValueRef>{first});
        const Value produced{ValueId{43}, spec, producer};
        expect(!produced.is_leaf(), "the produced constructor creates a non-leaf value");
        expect(produced.producer() == producer.get(), "a produced value exposes its producer node");
        expect(produced.producer_ref().get() == producer.get(),
               "a produced value retains an owning producer reference");

        expect_throws<std::invalid_argument>(
            [&spec]
            {
                const Value invalid{ValueId{44}, spec, NodeRef{}};
                (void)invalid;
            },
            "produced value construction rejects a null producer");

        expect_throws<std::invalid_argument>(
            []
            {
                const Node invalid{NodeId{60}, std::unique_ptr<Primitive>{}, {}};
                (void)invalid;
            },
            "node construction rejects a null primitive");
        expect_throws<std::invalid_argument>(
            [&first]
            {
                const Node invalid{NodeId{61}, std::make_unique<IdentitySpecPrimitive>(),
                                   std::vector<ValueRef>{first, ValueRef{}}};
                (void)invalid;
            },
            "node construction rejects null inputs");
    }
}
