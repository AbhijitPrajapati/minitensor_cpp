#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <array>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/leaf.hpp"
#include "tensor/graph/node.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/primitives/creation/full.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"
#include "tensor/tensor_access.hpp"
#include "tensor/support/test_buffer.hpp"
#include "tensor/support/test_primitive.hpp"

namespace minitensor::test
{
	TEST(ValueTest, RepresentsUnmaterializedLeaves)
	{
		const detail::TensorSpec spec{ Shape{ 2, 3 }, DType::Float32, Device::cpu(4) };
		const detail::Value value{ spec };

		EXPECT_EQ(value.spec(), spec);
		EXPECT_TRUE(value.is_leaf());
		EXPECT_EQ(value.producer(), nullptr);
		EXPECT_FALSE(value.producer_ref());
		EXPECT_EQ(value.materialization(), nullptr);
	}

	TEST(NodeTest, OwnsPrimitiveAndOrderedInputs)
	{
		const detail::TensorSpec spec{ Shape{ 2, 3 }, DType::Float32, Device::cpu() };
		const auto first = std::make_shared<detail::Value>(spec);
		const auto second = std::make_shared<detail::Value>(spec);
		auto primitive = std::make_unique<IdentitySpecPrimitive>();
		const detail::Primitive* address = primitive.get();
		const detail::Node node{ std::move(primitive), { first, second } };

		EXPECT_EQ(primitive, nullptr);
		EXPECT_EQ(&node.primitive(), address);
		EXPECT_EQ(node.primitive().name(), "test_identity");
		EXPECT_TRUE(node.primitive().requires_kernel_support());
		EXPECT_FALSE(node.primitive().try_derive_shared_layout(
			spec, detail::Layout::contiguous(spec.shape), spec));
		ASSERT_EQ(node.inputs().size(), 2);
		EXPECT_EQ(node.inputs()[0], first);
		EXPECT_EQ(node.inputs()[1], second);
	}

	TEST(NodeTest, ValidatesConstruction)
	{
		const detail::TensorSpec spec{ Shape{ 2 }, DType::Float32, Device::cpu() };
		const auto input = std::make_shared<detail::Value>(spec);
		EXPECT_THROW(
			(void)(detail::Node{ std::unique_ptr<detail::Primitive>{}, {} }),
			std::invalid_argument);
		EXPECT_THROW(
			(void)(detail::Node{
				std::make_unique<IdentitySpecPrimitive>(),
				{ input, detail::ValueRef{} } }),
			std::invalid_argument);
	}

	TEST(ValueTest, RepresentsProducedValuesAndValidatesProducer)
	{
		const detail::TensorSpec spec{ Shape{ 2 }, DType::Float32, Device::cpu() };
		const auto input = std::make_shared<detail::Value>(spec);
		const auto producer = std::make_shared<detail::Node>(
			std::make_unique<IdentitySpecPrimitive>(),
			std::vector<detail::ValueRef>{ input });
		const detail::Value value{ spec, producer };

		EXPECT_FALSE(value.is_leaf());
		EXPECT_EQ(value.producer(), producer.get());
		EXPECT_EQ(value.producer_ref(), producer);
		EXPECT_THROW(
			(void)(detail::Value{ spec, detail::NodeRef{} }),
			std::invalid_argument);
	}

	TEST(LeafTest, ProducesAProducerFreeValueWithoutCopyingStorage)
	{
		const detail::TensorSpec spec{
			Shape{ 2, 3 }, DType::Float32, Device::cpu() };
		const detail::BufferRef buffer = make_test_buffer(6 * sizeof(float));
		const detail::Layout layout = detail::Layout::contiguous(spec.shape);
		const detail::ValueRef input = detail::make_materialized_leaf(
			spec, detail::Materialization{ buffer, layout });

		EXPECT_TRUE(input->is_leaf());
		EXPECT_EQ(detail::leafify_materialized(input), input);

		detail::NodeRef producer = std::make_shared<detail::Node>(
			std::make_unique<IdentitySpecPrimitive>(),
			std::vector<detail::ValueRef>{ input });
		std::weak_ptr<const detail::Node> weak_producer = producer;
		detail::ValueRef produced = std::make_shared<detail::Value>(spec, producer);
		produced->materialize(detail::Materialization{ buffer, layout });

		const detail::ValueRef leafified = detail::leafify_materialized(produced);

		EXPECT_NE(leafified, produced);
		EXPECT_TRUE(leafified->is_leaf());
		EXPECT_EQ(leafified->producer(), nullptr);
		EXPECT_EQ(leafified->spec(), spec);
		ASSERT_NE(leafified->materialization(), nullptr);
		EXPECT_EQ(leafified->materialization()->buffer_ref(), buffer);
		EXPECT_EQ(leafified->materialization()->layout(), layout);

		produced.reset();
		producer.reset();
		EXPECT_TRUE(weak_producer.expired());
		EXPECT_EQ(leafified->materialization()->buffer_ref(), buffer);
	}

	TEST(LeafTest, RejectsNullAndUnmaterializedValues)
	{
		const detail::TensorSpec spec{
			Shape{ 2 }, DType::Float32, Device::cpu() };
		const detail::ValueRef unmaterialized =
			std::make_shared<detail::Value>(spec);

		EXPECT_THROW(
			(void)detail::leafify_materialized(detail::ValueRef{}),
			std::invalid_argument);
		EXPECT_THROW(
			(void)detail::leafify_materialized(unmaterialized),
			std::logic_error);
	}

	TEST(TensorAccessTest, ConvertsBetweenHandlesAndValues)
	{
		const auto value = std::make_shared<detail::Value>(
			detail::TensorSpec{ Shape{ 2 }, DType::Float32, Device::cpu() });
		const Tensor tensor = detail::TensorAccess::make(value);
		EXPECT_EQ(detail::TensorAccess::value(tensor), value);
		EXPECT_THROW(
			(void)detail::TensorAccess::make(detail::ValueRef{}),
			std::invalid_argument);
	}

	TEST(ApplyOperationTest, InfersALazyProducedValue)
	{
		const detail::TensorSpec spec{ Shape{ 2, 3 }, DType::Float32, Device::cpu(2) };
		const auto input = std::make_shared<detail::Value>(spec);
		auto primitive = std::make_unique<IdentitySpecPrimitive>();
		const detail::Primitive* primitive_address = primitive.get();
		const std::array<detail::ValueRef, 1> inputs{ input };

		const detail::ValueRef output = detail::apply_operation(std::move(primitive), inputs);
		ASSERT_NE(output, nullptr);
		EXPECT_EQ(primitive, nullptr);
		EXPECT_NE(output, input);
		EXPECT_EQ(output->spec(), spec);
		EXPECT_EQ(output->materialization(), nullptr);
		ASSERT_NE(output->producer(), nullptr);
		EXPECT_EQ(&output->producer()->primitive(), primitive_address);
		ASSERT_EQ(output->producer()->inputs().size(), 1);
		EXPECT_EQ(output->producer()->inputs().front(), input);
	}

	TEST(ApplyOperationTest, SupportsZeroInputPrimitives)
	{
		const detail::TensorSpec spec{ Shape{ 4, 1 }, DType::Float32, Device::cpu(3) };
		const detail::ValueRef output = detail::apply_operation(
			std::make_unique<detail::FullPrimitive>(spec, 7.25F),
			std::span<const detail::ValueRef>{});

		EXPECT_EQ(output->spec(), spec);
		const auto* primitive = dynamic_cast<const detail::FullPrimitive*>(
			&output->producer()->primitive());
		ASSERT_NE(primitive, nullptr);
		EXPECT_FLOAT_EQ(primitive->fill_value(), 7.25F);
	}

	TEST(ApplyOperationTest, ValidatesPrimitiveAndInputs)
	{
		const detail::TensorSpec spec{ Shape{ 2 }, DType::Float32, Device::cpu() };
		const auto input = std::make_shared<detail::Value>(spec);
		const std::array<detail::ValueRef, 1> inputs{ input };
		const std::array<detail::ValueRef, 1> null_inputs{ detail::ValueRef{} };

		EXPECT_THROW(
			(void)detail::apply_operation(
				std::unique_ptr<detail::Primitive>{}, inputs),
			std::invalid_argument);
		EXPECT_THROW(
			(void)detail::apply_operation(
				std::make_unique<IdentitySpecPrimitive>(), null_inputs),
			std::invalid_argument);
		EXPECT_THROW(
			(void)detail::apply_operation(
				std::make_unique<IdentitySpecPrimitive>(),
				std::span<const detail::ValueRef>{}),
			std::invalid_argument);
	}
}
