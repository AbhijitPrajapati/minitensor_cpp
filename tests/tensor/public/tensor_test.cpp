#include <minitensor/ops/creation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <utility>

#include <gtest/gtest.h>

namespace minitensor::test
{
	TEST(TensorTest, ExposesGraphVisibleMetadata)
	{
		const Tensor tensor = full(
			Shape{ 2, 3 }, 1.0F,
			TensorOptions{ DType::Float32, Device::cpu(4) });

		EXPECT_EQ(tensor.shape(), (Shape{ 2, 3 }));
		EXPECT_EQ(tensor.rank(), 2);
		EXPECT_EQ(tensor.numel(), 6);
		EXPECT_EQ(tensor.dtype(), DType::Float32);
		EXPECT_EQ(tensor.device(), Device::cpu(4));
	}

	TEST(TensorTest, HasSharedHandleValueSemantics)
	{
		const Tensor original = full(Shape{ 2, 3 }, 2.0F);
		const Tensor copied = original;
		EXPECT_EQ(copied.shape(), original.shape());

		Tensor assigned = full(Shape{}, 0.0F);
		assigned = original;
		EXPECT_EQ(assigned.shape(), original.shape());

		Tensor moved = std::move(assigned);
		EXPECT_EQ(moved.shape(), original.shape());
	}
}
