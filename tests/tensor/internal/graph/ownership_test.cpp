#include <minitensor/types.hpp>

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/node.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/support/test_primitive.hpp"

namespace minitensor::test
{
	TEST(GraphOwnershipTest, ProducedValueKeepsProducerAlive)
	{
		const detail::TensorSpec spec{ Shape{ 2 }, DType::Float32, Device::cpu() };
		detail::NodeRef producer = std::make_shared<detail::Node>(
			std::make_unique<IdentitySpecPrimitive>(),
			std::vector<detail::ValueRef>{});
		const std::weak_ptr<const detail::Node> weak_producer{ producer };

		{
			const auto output = std::make_shared<detail::Value>(spec, producer);
			producer.reset();
			EXPECT_FALSE(weak_producer.expired());
		}
		EXPECT_TRUE(weak_producer.expired());
	}

	TEST(GraphOwnershipTest, NodeKeepsInputsAndPrimitiveAlive)
	{
		const detail::TensorSpec spec{ Shape{ 2 }, DType::Float32, Device::cpu() };
		detail::ValueRef input = std::make_shared<detail::Value>(spec);
		bool primitive_destroyed = false;
		auto primitive = std::make_unique<DestructionTrackedPrimitive>(primitive_destroyed);
		const std::weak_ptr<detail::Value> weak_input{ input };
		detail::NodeRef node = std::make_shared<detail::Node>(
			std::move(primitive), std::vector<detail::ValueRef>{ input });

		input.reset();
		EXPECT_FALSE(weak_input.expired());
		EXPECT_FALSE(primitive_destroyed);

		node.reset();
		EXPECT_TRUE(weak_input.expired());
		EXPECT_TRUE(primitive_destroyed);
	}

	TEST(GraphOwnershipTest, FinalHandleOwnsAndReleasesTheTransitiveGraph)
	{
		const detail::TensorSpec spec{ Shape{ 2 }, DType::Float32, Device::cpu() };
		detail::ValueRef leaf = std::make_shared<detail::Value>(spec);
		bool first_destroyed = false;
		bool second_destroyed = false;

		std::array<detail::ValueRef, 1> first_inputs{ leaf };
		detail::ValueRef intermediate = detail::apply_operation(
			std::make_unique<DestructionTrackedPrimitive>(first_destroyed),
			first_inputs);
		std::array<detail::ValueRef, 1> second_inputs{ intermediate };
		detail::ValueRef output = detail::apply_operation(
			std::make_unique<DestructionTrackedPrimitive>(second_destroyed),
			second_inputs);

		detail::NodeRef first_node = intermediate->producer_ref();
		detail::NodeRef second_node = output->producer_ref();
		const std::weak_ptr<detail::Value> weak_leaf{ leaf };
		const std::weak_ptr<detail::Value> weak_intermediate{ intermediate };
		const std::weak_ptr<const detail::Node> weak_first_node{ first_node };
		const std::weak_ptr<const detail::Node> weak_second_node{ second_node };

		leaf.reset();
		intermediate.reset();
		first_inputs.front().reset();
		second_inputs.front().reset();
		first_node.reset();
		second_node.reset();

		EXPECT_FALSE(weak_leaf.expired());
		EXPECT_FALSE(weak_intermediate.expired());
		EXPECT_FALSE(weak_first_node.expired());
		EXPECT_FALSE(weak_second_node.expired());
		EXPECT_FALSE(first_destroyed);
		EXPECT_FALSE(second_destroyed);

		output.reset();
		EXPECT_TRUE(weak_leaf.expired());
		EXPECT_TRUE(weak_intermediate.expired());
		EXPECT_TRUE(weak_first_node.expired());
		EXPECT_TRUE(weak_second_node.expired());
		EXPECT_TRUE(first_destroyed);
		EXPECT_TRUE(second_destroyed);
	}
}
