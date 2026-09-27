#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor
{
	inline void PrintTo(const Shape& shape, std::ostream* stream)
	{
		*stream << "Shape{";
		for (std::size_t axis = 0; axis < shape.rank(); ++axis)
		{
			if (axis != 0)
			{
				*stream << ", ";
			}
			*stream << shape[axis];
		}
		*stream << '}';
	}

	inline void PrintTo(const Device& device, std::ostream* stream)
	{
		*stream << "Device{" << static_cast<int>(device.type()) << ", "
			<< device.index() << '}';
	}
}

namespace minitensor::test
{
	inline void expect_values_near(
		std::span<const float> actual,
		std::span<const float> expected,
		float absolute_tolerance = 1.0E-6F,
		float relative_tolerance = 1.0E-5F)
	{
		ASSERT_EQ(actual.size(), expected.size());
		for (std::size_t index = 0; index < actual.size(); ++index)
		{
			const float tolerance = absolute_tolerance +
				relative_tolerance * std::fabs(expected[index]);
			EXPECT_NEAR(actual[index], expected[index], tolerance)
				<< "at logical element " << index;
		}
	}

	inline void expect_tensor_near(
		const Tensor& tensor,
		std::span<const float> expected,
		float absolute_tolerance = 1.0E-6F,
		float relative_tolerance = 1.0E-5F)
	{
		expect_values_near(
			to_vector(tensor), expected, absolute_tolerance, relative_tolerance);
	}

	inline void expect_tensor_eq(
		const Tensor& tensor,
		std::span<const float> expected)
	{
		EXPECT_EQ(to_vector(tensor), std::vector<float>(expected.begin(), expected.end()));
	}
}
