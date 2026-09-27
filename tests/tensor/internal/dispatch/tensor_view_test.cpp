#include <minitensor/types.hpp>

#include <stdexcept>

#include <gtest/gtest.h>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"
#include "tensor/support/test_buffer.hpp"

namespace minitensor::test
{
	TEST(TensorViewTest, ExposesReadOnlyTensorStorage)
	{
		const detail::TensorSpec spec{ Shape{ 2, 2 }, DType::Float32, Device::cpu(2) };
		const detail::Layout layout{ { 3, 1 }, 1 };
		const detail::BufferRef buffer = make_test_buffer(6 * sizeof(float), spec.device);
		const detail::Materialization storage{ buffer, layout };
		const detail::TensorView view{ spec, storage };

		EXPECT_EQ(view.shape(), spec.shape);
		EXPECT_EQ(view.dtype(), spec.dtype);
		EXPECT_EQ(view.device(), spec.device);
		EXPECT_EQ(view.layout(), layout);
		EXPECT_EQ(&view.buffer(), buffer.get());
	}

	TEST(TensorViewTest, ExposesMutableTensorStorage)
	{
		const detail::TensorSpec spec{ Shape{ 2, 2 }, DType::Float32, Device::cpu(2) };
		const detail::Layout layout{ { 3, 1 }, 1 };
		const detail::BufferRef buffer = make_test_buffer(6 * sizeof(float), spec.device);
		const detail::Materialization storage{ buffer, layout };
		const detail::MutableTensorView view{ spec, storage };

		EXPECT_EQ(view.shape(), spec.shape);
		EXPECT_EQ(view.dtype(), spec.dtype);
		EXPECT_EQ(view.device(), spec.device);
		EXPECT_EQ(view.layout(), layout);
		EXPECT_EQ(&view.buffer(), buffer.get());
	}

	TEST(TensorViewTest, ValidatesMaterializations)
	{
		const detail::TensorSpec spec{ Shape{ 2, 2 }, DType::Float32, Device::cpu() };
		const detail::Materialization wrong_rank{
			make_test_buffer(4 * sizeof(float)), detail::Layout{ { 1 } } };
		EXPECT_THROW((void)detail::TensorView(spec, wrong_rank), std::invalid_argument);

		const detail::Materialization wrong_device{
			make_test_buffer(4 * sizeof(float), Device::cpu(1)),
			detail::Layout::contiguous(spec.shape) };
		EXPECT_THROW(
			(void)detail::MutableTensorView(spec, wrong_device),
			std::invalid_argument);
	}
}
