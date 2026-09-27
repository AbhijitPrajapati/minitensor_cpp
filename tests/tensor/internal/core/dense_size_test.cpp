#include <minitensor/types.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "tensor/core/dense_size.hpp"
#include "tensor/core/tensor_spec.hpp"

namespace minitensor::test
{
	TEST(DenseSizeTest, ComputesScalarAndTensorStorage)
	{
		EXPECT_EQ(
			detail::dense_size_bytes(
				detail::TensorSpec{ Shape{}, DType::Float32, Device::cpu() }),
			sizeof(float));
		EXPECT_EQ(
			detail::dense_size_bytes(
				detail::TensorSpec{ Shape{ 2, 3 }, DType::Float32, Device::cpu(4) }),
			6 * sizeof(float));
	}

	TEST(DenseSizeTest, EmptyTensorsRequireNoStorage)
	{
		EXPECT_EQ(
			detail::dense_size_bytes(
				detail::TensorSpec{ Shape{ 2, 0, 3 }, DType::Float32, Device::cpu() }),
			0);
	}

	TEST(DenseSizeTest, RejectsByteCountOverflow)
	{
		constexpr auto extent = static_cast<Extent>(
			std::numeric_limits<std::size_t>::max() / sizeof(float) + 1);
		const detail::TensorSpec spec{ Shape{ extent }, DType::Float32, Device::cpu() };
		EXPECT_THROW((void)detail::dense_size_bytes(spec), std::overflow_error);
	}
}
