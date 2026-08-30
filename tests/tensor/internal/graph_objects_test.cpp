#include <minitensor/types.hpp>

#include <array>
#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/ids.hpp"
#include "tensor/graph/node.hpp"
#include "tensor/graph/origin.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/tensor_access.hpp"

#include "../support/test.hpp"
#include "../support/test_primitive.hpp"

namespace minitensor::test
{
    void run_graph_objects_test()
    {
        using detail::LeafOrigin;
        using detail::Node;
        using detail::NodeId;
        using detail::PrimitiveRef;
        using detail::ProducedOrigin;
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
        const Value leaf{ValueId{40}, spec, LeafOrigin{}};
        expect(leaf.id() == ValueId{40}, "a value preserves its id");
        expect(leaf.spec() == spec, "a value preserves its tensor specification");
        expect(std::holds_alternative<LeafOrigin>(leaf.origin()), "a leaf value preserves its origin");
        expect(leaf.materialization() == nullptr, "a newly constructed value has no materialization");

        const ValueRef first = std::make_shared<Value>(ValueId{41}, spec, LeafOrigin{});
        const ValueRef second = std::make_shared<Value>(ValueId{42}, spec, LeafOrigin{});
        const auto primitive = std::make_shared<IdentitySpecPrimitive>();
        const Node node{NodeId{50}, primitive, {first, second}};

        expect(node.id() == NodeId{50}, "a node preserves its id");
        expect(node.primitive().name() == "test_identity", "a node exposes its primitive");
        expect(node.primitive_ref().get() == primitive.get(), "a node retains the supplied primitive object");
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

        const auto producer = std::make_shared<Node>(NodeId{51}, primitive, std::vector<ValueRef>{first});
        const Value produced{ValueId{43}, spec, ProducedOrigin{producer}};
        const ProducedOrigin *produced_origin = std::get_if<ProducedOrigin>(&produced.origin());
        expect(produced_origin != nullptr, "a produced value preserves its origin kind");
        expect(produced_origin->node.get() == producer.get(), "a produced value retains its producer node");

        expect_throws<std::invalid_argument>(
            []
            {
                const Node invalid{NodeId{60}, PrimitiveRef{}, {}};
                (void)invalid;
            },
            "node construction rejects a null primitive");
        expect_throws<std::invalid_argument>(
            [&primitive, &first]
            {
                const Node invalid{NodeId{61}, primitive, std::vector<ValueRef>{first, ValueRef{}}};
                (void)invalid;
            },
            "node construction rejects null inputs");
    }
}
