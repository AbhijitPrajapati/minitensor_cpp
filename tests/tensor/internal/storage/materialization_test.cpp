#include <minitensor/types.hpp>

#include <stdexcept>

#include <gtest/gtest.h>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"
#include "tensor/support/test_buffer.hpp"

namespace minitensor::test
{
	TEST(MaterializationTest, OwnsBufferAndLayout)
	{
		const detail::BufferRef buffer = make_test_buffer(6 * sizeof(float), Device::cpu(2));
		const detail::Layout layout = detail::Layout::contiguous(Shape{ 2, 3 });
		const detail::Materialization materialization{ buffer, layout };

		EXPECT_EQ(materialization.buffer_ref(), buffer);
		EXPECT_EQ(materialization.layout(), layout);
		EXPECT_NO_THROW(materialization.validate(
			detail::TensorSpec{ Shape{ 2, 3 }, DType::Float32, Device::cpu(2) }));
	}

	TEST(MaterializationTest, RejectsNullBuffers)
	{
		EXPECT_THROW(
			(void)(detail::Materialization{ detail::BufferRef{}, detail::Layout{} }),
			std::invalid_argument);
	}

	TEST(MaterializationTest, SupportsNegativeStridesAndEmptyLayouts)
	{
		const detail::Materialization reversed{
			make_test_buffer(6 * sizeof(float)), detail::Layout{ { -3, 1 }, 3 } };
		EXPECT_NO_THROW(reversed.validate(
			detail::TensorSpec{ Shape{ 2, 3 }, DType::Float32, Device::cpu() }));

		const detail::Materialization empty{
			make_test_buffer(0), detail::Layout{ { 9, 1 }, 50 } };
		EXPECT_NO_THROW(empty.validate(
			detail::TensorSpec{ Shape{ 0, 3 }, DType::Float32, Device::cpu() }));
	}

	TEST(MaterializationTest, ValidatesDeviceRankAndReachableStorage)
	{
		const detail::Materialization wrong_device{
			make_test_buffer(2 * sizeof(float), Device::cpu(1)),
			detail::Layout::contiguous(Shape{ 2 }) };
		EXPECT_THROW(
			wrong_device.validate(
				detail::TensorSpec{ Shape{ 2 }, DType::Float32, Device::cpu() }),
			std::invalid_argument);

		const detail::Materialization wrong_rank{
			make_test_buffer(6 * sizeof(float)), detail::Layout{ { 1 } } };
		EXPECT_THROW(
			wrong_rank.validate(
				detail::TensorSpec{ Shape{ 2, 3 }, DType::Float32, Device::cpu() }),
			std::invalid_argument);

		const detail::Materialization too_small{
			make_test_buffer(5 * sizeof(float)),
			detail::Layout::contiguous(Shape{ 2, 3 }) };
		EXPECT_THROW(
			too_small.validate(
				detail::TensorSpec{ Shape{ 2, 3 }, DType::Float32, Device::cpu() }),
			std::invalid_argument);

		const detail::Materialization before_start{
			make_test_buffer(2 * sizeof(float)), detail::Layout{ { -1 } } };
		EXPECT_THROW(
			before_start.validate(
				detail::TensorSpec{ Shape{ 2 }, DType::Float32, Device::cpu() }),
			std::invalid_argument);
	}

	TEST(ValueMaterializationTest, InstallsStorageExactlyOnce)
	{
		const detail::TensorSpec spec{ Shape{ 2, 3 }, DType::Float32, Device::cpu() };
		const detail::BufferRef buffer = make_test_buffer(6 * sizeof(float));
		const detail::Layout layout = detail::Layout::contiguous(spec.shape);
		detail::Value value{ spec };

		EXPECT_EQ(value.materialization(), nullptr);
		value.materialize(detail::Materialization{ buffer, layout });
		ASSERT_NE(value.materialization(), nullptr);
		EXPECT_EQ(value.materialization()->buffer_ref(), buffer);
		EXPECT_THROW(
			value.materialize(detail::Materialization{ buffer, layout }),
			std::logic_error);
	}

	TEST(ValueMaterializationTest, FailedValidationLeavesValueUnmaterialized)
	{
		detail::Value value{
			detail::TensorSpec{ Shape{ 2 }, DType::Float32, Device::cpu() } };
		EXPECT_THROW(
			value.materialize(detail::Materialization{
				make_test_buffer(sizeof(float)),
				detail::Layout::contiguous(Shape{ 2 }) }),
			std::invalid_argument);
		EXPECT_EQ(value.materialization(), nullptr);
	}
}
