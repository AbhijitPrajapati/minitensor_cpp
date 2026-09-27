#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/linalg.hpp>
#include <minitensor/ops/manipulation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <array>
#include <stdexcept>

#include <gtest/gtest.h>

#include "tensor/support/tensor_assertions.hpp"

namespace minitensor::test
{
	TEST(MatmulTest, InfersVectorMatrixAndBatchedShapes)
	{
		EXPECT_EQ(
			matmul(full(Shape{ 3 }, 1.0F), full(Shape{ 3 }, 1.0F)).shape(),
			Shape{});
		EXPECT_EQ(
			matmul(full(Shape{ 2, 3 }, 1.0F), full(Shape{ 3 }, 1.0F)).shape(),
			(Shape{ 2 }));
		EXPECT_EQ(
			matmul(full(Shape{ 3 }, 1.0F), full(Shape{ 3, 4 }, 1.0F)).shape(),
			(Shape{ 4 }));
		EXPECT_EQ(
			matmul(full(Shape{ 2, 3 }, 1.0F), full(Shape{ 3, 4 }, 1.0F)).shape(),
			(Shape{ 2, 4 }));
		EXPECT_EQ(
			matmul(
				full(Shape{ 2, 1, 3, 4 }, 1.0F),
				full(Shape{ 5, 4, 6 }, 1.0F)).shape(),
			(Shape{ 2, 5, 3, 6 }));
	}

	TEST(MatmulTest, ComputesAllVectorAndMatrixForms)
	{
		const Tensor lhs_vector = from_data({ 1.0F, 2.0F, 3.0F }, Shape{ 3 });
		const Tensor rhs_vector = from_data({ 4.0F, 5.0F, 6.0F }, Shape{ 3 });
		EXPECT_FLOAT_EQ(item(matmul(lhs_vector, rhs_vector)), 32.0F);

		const Tensor matrix = from_data(
			{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Shape{ 2, 3 });
		const std::array<float, 2> matrix_vector{ -2.0F, -2.0F };
		expect_tensor_eq(
			matmul(matrix, from_data({ 1.0F, 0.0F, -1.0F }, Shape{ 3 })),
			matrix_vector);

		const std::array<float, 3> vector_matrix{ 9.0F, 12.0F, 15.0F };
		expect_tensor_eq(
			matmul(from_data({ 1.0F, 2.0F }, Shape{ 2 }), matrix),
			vector_matrix);

		const Tensor rhs_matrix = from_data(
			{ 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F }, Shape{ 3, 2 });
		const std::array<float, 4> matrix_matrix{ 58.0F, 64.0F, 139.0F, 154.0F };
		expect_tensor_eq(matmul(matrix, rhs_matrix), matrix_matrix);
	}

	TEST(MatmulTest, BroadcastsBatches)
	{
		const Tensor lhs = from_data(
			{ 1.0F, 2.0F, 3.0F, 4.0F }, Shape{ 2, 1, 1, 2 });
		const Tensor rhs = from_data(
			{ 10.0F, 1.0F, 2.0F, 3.0F, -1.0F, 5.0F }, Shape{ 3, 2, 1 });
		const std::array<float, 6> expected{
			12.0F, 8.0F, 9.0F,
			34.0F, 18.0F, 17.0F
		};
		expect_tensor_eq(matmul(lhs, rhs), expected);
	}

	TEST(MatmulTest, HonorsStridesInBothOperands)
	{
		const Tensor lhs = transpose(from_data(
			{ 1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F }, Shape{ 3, 2 }));
		const Tensor rhs = transpose(from_data(
			{ 7.0F, 9.0F, 11.0F, 8.0F, 10.0F, 12.0F }, Shape{ 2, 3 }));
		const std::array<float, 4> expected{ 58.0F, 64.0F, 139.0F, 154.0F };
		expect_tensor_eq(matmul(lhs, rhs), expected);
	}

	TEST(MatmulTest, HandlesEmptyContractionsAndBatches)
	{
		EXPECT_FLOAT_EQ(
			item(matmul(full(Shape{ 0 }, 1.0F), full(Shape{ 0 }, 1.0F))),
			0.0F);

		const std::array<float, 6> zero_matrix{};
		expect_tensor_eq(
			matmul(full(Shape{ 2, 0 }, 1.0F), full(Shape{ 0, 3 }, 1.0F)),
			zero_matrix);

		EXPECT_TRUE(to_vector(matmul(
			full(Shape{ 0, 2, 3 }, 1.0F),
			full(Shape{ 3, 4 }, 1.0F))).empty());
	}

	TEST(MatmulTest, ValidatesOperands)
	{
		EXPECT_THROW(
			(void)matmul(full(Shape{}, 1.0F), full(Shape{ 1 }, 1.0F)),
			std::invalid_argument);
		EXPECT_THROW(
			(void)matmul(full(Shape{ 2, 3 }, 1.0F), full(Shape{ 2, 4 }, 1.0F)),
			std::invalid_argument);
		EXPECT_THROW(
			(void)matmul(
				full(Shape{ 2, 3, 4 }, 1.0F),
				full(Shape{ 5, 4, 6 }, 1.0F)),
			std::invalid_argument);

		const Tensor lhs = full(
			Shape{ 2, 2 }, 1.0F,
			TensorOptions{ DType::Float32, Device::cpu(0) });
		const Tensor rhs = full(
			Shape{ 2, 2 }, 1.0F,
			TensorOptions{ DType::Float32, Device::cpu(1) });
		EXPECT_THROW((void)matmul(lhs, rhs), std::invalid_argument);
	}
}
