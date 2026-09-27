#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/manipulation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <array>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>

#include <gtest/gtest.h>

#include "tensor/support/tensor_assertions.hpp"

namespace minitensor::test
{
	TEST(PermuteTest, InfersShapeAndPreservesLogicalValues)
	{
		const std::array<float, 6> values{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
		const Tensor matrix = from_data(values, Shape{ 2, 3 });
		const Tensor result = permute(matrix, { -1, 0 });
		const std::array<float, 6> expected{ 1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F };

		EXPECT_EQ(result.shape(), (Shape{ 3, 2 }));
		EXPECT_EQ(result.dtype(), matrix.dtype());
		EXPECT_EQ(result.device(), matrix.device());
		expect_tensor_eq(result, expected);
	}

	TEST(PermuteTest, SupportsScalars)
	{
		const Tensor scalar = full(Shape{}, 2.0F);
		EXPECT_TRUE(permute(scalar, std::span<const Axis>{}).shape().is_scalar());
	}

	TEST(PermuteTest, ValidatesPermutation)
	{
		const Tensor matrix = full(Shape{ 2, 3 }, 1.0F);
		EXPECT_THROW((void)permute(matrix, { 0 }), std::invalid_argument);
		EXPECT_THROW((void)permute(matrix, { 0, 0 }), std::invalid_argument);
		EXPECT_THROW((void)permute(matrix, { 0, 2 }), std::out_of_range);
	}

	TEST(TransposeTest, SupportsDefaultAndSelectedAxes)
	{
		const Tensor volume = full(Shape{ 2, 3, 4 }, 1.0F);
		EXPECT_EQ(transpose(volume).shape(), (Shape{ 4, 3, 2 }));
		EXPECT_EQ(transpose(volume, 0, -1).shape(), (Shape{ 4, 3, 2 }));
		EXPECT_EQ(transpose(volume, 1, 1).shape(), volume.shape());

		const Tensor scalar = full(Shape{}, 1.0F);
		EXPECT_TRUE(transpose(scalar).shape().is_scalar());
		EXPECT_THROW((void)transpose(volume, 0, 3), std::out_of_range);
	}

	TEST(ContiguousTest, CopiesOnlyWhenLogicalLayoutRequiresIt)
	{
		const std::array<float, 6> values{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
		const Tensor permuted = transpose(from_data(values, Shape{ 2, 3 }));
		const std::array<float, 6> expected{ 1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F };

		const Tensor result = contiguous(permuted);
		EXPECT_EQ(result.shape(), permuted.shape());
		expect_tensor_eq(result, expected);
	}

	TEST(ReshapeTest, PreservesLogicalOrderForContiguousAndStridedInputs)
	{
		const std::array<float, 6> values{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
		const Tensor matrix = from_data(values, Shape{ 2, 3 });
		const Tensor contiguous_result = reshape(matrix, Shape{ 3, 2 });
		EXPECT_EQ(contiguous_result.shape(), (Shape{ 3, 2 }));
		expect_tensor_eq(contiguous_result, values);

		const Tensor permuted = transpose(matrix);
		const Tensor strided_result = reshape(permuted, Shape{ 2, 3 });
		const std::array<float, 6> expected{ 1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F };
		expect_tensor_eq(strided_result, expected);

		EXPECT_EQ(reshape(full(Shape{}, 1.0F), Shape{ 1 }).shape(), (Shape{ 1 }));
		EXPECT_EQ(reshape(full(Shape{ 2, 0, 3 }, 1.0F), Shape{ 0, 6 }).shape(), (Shape{ 0, 6 }));
		EXPECT_THROW((void)reshape(matrix, Shape{ 5 }), std::invalid_argument);
	}

	TEST(FlattenTest, NormalizesAndValidatesAxisRanges)
	{
		const Tensor volume = full(Shape{ 2, 3, 4 }, 1.0F);
		EXPECT_EQ(flatten(volume).shape(), (Shape{ 24 }));
		EXPECT_EQ(flatten(volume, 1).shape(), (Shape{ 2, 12 }));
		EXPECT_EQ(flatten(volume, -2, -1).shape(), (Shape{ 2, 12 }));
		EXPECT_EQ(flatten(volume, 1, 1).shape(), volume.shape());
		EXPECT_EQ(flatten(full(Shape{}, 1.0F)).shape(), (Shape{ 1 }));
		EXPECT_EQ(flatten(full(Shape{ 2, 0, 3 }, 1.0F)).shape(), (Shape{ 0 }));

		EXPECT_THROW((void)flatten(volume, 2, 1), std::invalid_argument);
		EXPECT_THROW((void)flatten(volume, 0, 3), std::out_of_range);
	}

	TEST(SqueezeTest, RemovesRequestedSingletonDimensions)
	{
		const Tensor input = full(Shape{ 1, 2, 1, 3 }, 1.0F);
		EXPECT_EQ(squeeze(input).shape(), (Shape{ 2, 3 }));
		EXPECT_EQ(squeeze(input, 0).shape(), (Shape{ 2, 1, 3 }));
		EXPECT_EQ(squeeze(input, { 0, -2 }).shape(), (Shape{ 2, 3 }));
		EXPECT_EQ(squeeze(input, std::span<const Axis>{}).shape(), input.shape());

		EXPECT_THROW((void)squeeze(input, 1), std::invalid_argument);
		EXPECT_THROW((void)squeeze(input, { 0, -4 }), std::invalid_argument);
	}

	TEST(UnsqueezeTest, InsertsSingletonDimensions)
	{
		const Tensor matrix = full(Shape{ 2, 3 }, 1.0F);
		EXPECT_EQ(unsqueeze(matrix, 0).shape(), (Shape{ 1, 2, 3 }));
		EXPECT_EQ(unsqueeze(matrix, -1).shape(), (Shape{ 2, 3, 1 }));
		EXPECT_EQ(unsqueeze(full(Shape{}, 1.0F), 0).shape(), (Shape{ 1 }));
		EXPECT_THROW((void)unsqueeze(matrix, 3), std::out_of_range);
	}

	TEST(BroadcastToTest, ExpandsSingletonAndLeadingDimensions)
	{
		const std::array<float, 3> values{ 7.0F, 8.0F, 9.0F };
		const Tensor row = from_data(values, Shape{ 1, 3 });
		const Tensor result = broadcast_to(row, Shape{ 2, 3 });
		const std::array<float, 6> expected{ 7.0F, 8.0F, 9.0F, 7.0F, 8.0F, 9.0F };

		EXPECT_EQ(result.shape(), (Shape{ 2, 3 }));
		expect_tensor_eq(result, expected);
		EXPECT_EQ(broadcast_to(full(Shape{}, 1.0F), Shape{ 2, 3 }).shape(), (Shape{ 2, 3 }));

		EXPECT_THROW((void)broadcast_to(row, Shape{ 2, 2 }), std::invalid_argument);
		EXPECT_THROW((void)broadcast_to(row, Shape{ 3 }), std::invalid_argument);
	}

	TEST(ConcatenateTest, JoinsInputsInLogicalOrder)
	{
		const Tensor first = from_data({ 1.0F, 4.0F }, Shape{ 2, 1 });
		const Tensor empty = from_data(std::span<const float>{}, Shape{ 2, 0 });
		const Tensor second = from_data({ 2.0F, 3.0F, 5.0F, 6.0F }, Shape{ 2, 2 });
		const std::array<Tensor, 3> inputs{ first, empty, second };
		const Tensor result = concatenate(inputs, -1);
		const std::array<float, 6> expected{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };

		EXPECT_EQ(result.shape(), (Shape{ 2, 3 }));
		expect_tensor_eq(result, expected);
		expect_tensor_eq(concatenate({ second }), std::array<float, 4>{ 2.0F, 3.0F, 5.0F, 6.0F });
	}

	TEST(ConcatenateTest, ValidatesInputs)
	{
		EXPECT_THROW((void)concatenate(std::span<const Tensor>{}), std::invalid_argument);

		const Tensor scalar = full(Shape{}, 1.0F);
		EXPECT_THROW((void)concatenate({ scalar, scalar }), std::out_of_range);

		const Tensor matrix = full(Shape{ 2, 3 }, 1.0F);
		EXPECT_THROW(
			(void)concatenate({ matrix, full(Shape{ 3, 3 }, 1.0F) }, 1),
			std::invalid_argument);
		EXPECT_THROW(
			(void)concatenate({ matrix, full(Shape{ 2, 3, 1 }, 1.0F) }, 1),
			std::invalid_argument);
		EXPECT_THROW(
			(void)concatenate({
				matrix,
				full(
					Shape{ 2, 3 }, 1.0F,
					TensorOptions{ DType::Float32, Device::cpu(1) }) }, 1),
			std::invalid_argument);
	}

	TEST(ConcatenateTest, RejectsExtentOverflow)
	{
		const Extent maximum = std::numeric_limits<Extent>::max();
		const Tensor lhs = full(Shape{ 1, maximum }, 0.0F);
		const Tensor rhs = full(Shape{ 1, 1 }, 0.0F);
		EXPECT_THROW((void)concatenate({ lhs, rhs }, 1), std::overflow_error);
	}

	TEST(SliceTest, SupportsStridesClippingAndReverseTraversal)
	{
		const std::array<float, 6> values{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
		const Tensor matrix = from_data(values, Shape{ 2, 3 });

		const std::array<float, 4> strided{ 1.0F, 3.0F, 4.0F, 6.0F };
		expect_tensor_eq(slice(matrix, 1, -100, 100, 2), strided);

		const std::array<float, 6> reversed{ 3.0F, 2.0F, 1.0F, 6.0F, 5.0F, 4.0F };
		expect_tensor_eq(
			slice(matrix, -1, std::nullopt, std::nullopt, -1),
			reversed);

		EXPECT_EQ(slice(matrix, 1, 2, 1).shape(), (Shape{ 2, 0 }));
		EXPECT_THROW(
			(void)slice(matrix, 0, std::nullopt, std::nullopt, 0),
			std::invalid_argument);
		EXPECT_THROW((void)slice(matrix, 2), std::out_of_range);
	}
}
