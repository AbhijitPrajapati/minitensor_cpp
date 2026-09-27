#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/manipulation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <array>
#include <span>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace minitensor::test
{
	TEST(DataTest, RoundTripsHostDataByValue)
	{
		std::array<float, 6> source{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
		const std::array<float, 6> expected = source;
		const Tensor tensor = from_data(source, Shape{ 2, 3 });

		EXPECT_EQ(tensor.shape(), (Shape{ 2, 3 }));
		EXPECT_EQ(tensor.dtype(), DType::Float32);
		EXPECT_EQ(tensor.device(), Device::cpu());

		source[0] = -100.0F;
		EXPECT_EQ(to_vector(tensor), std::vector<float>(expected.begin(), expected.end()));
	}

	TEST(DataTest, ReadsViewsInLogicalOrder)
	{
		const Tensor matrix = from_data(
			{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Shape{ 2, 3 });
		const Tensor transposed = transpose(matrix);
		const std::vector<float> expected{ 1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F };
		EXPECT_EQ(to_vector(transposed), expected);
	}

	TEST(DataTest, SupportsScalarsSingletonsAndEmptyTensors)
	{
		EXPECT_FLOAT_EQ(item(from_data({ -3.25F }, Shape{})), -3.25F);
		EXPECT_FLOAT_EQ(item(full(Shape{ 1, 1 }, 7.5F)), 7.5F);

		const Tensor empty = from_data(std::span<const float>{}, Shape{ 2, 0, 3 });
		EXPECT_TRUE(to_vector(empty).empty());
	}

	TEST(DataTest, ValidatesElementCounts)
	{
		const std::array<float, 2> values{ 1.0F, 2.0F };
		EXPECT_THROW((void)from_data(values, Shape{ 3 }), std::invalid_argument);
		EXPECT_THROW((void)item(full(Shape{ 2 }, 1.0F)), std::invalid_argument);
	}

	TEST(DataTest, ReportsUnavailableDevices)
	{
		const std::array<float, 1> values{ 1.0F };
		EXPECT_THROW(
			(void)from_data(
				values,
				Shape{ 1 },
				TensorOptions{ DType::Float32, Device::cpu(1) }),
			std::runtime_error);
	}
}
