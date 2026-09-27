#include <minitensor/autograd.hpp>
#include <minitensor/data.hpp>
#include <minitensor/ops/linalg.hpp>
#include <minitensor/ops/reduction.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <array>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/support/tensor_assertions.hpp"

namespace minitensor::test
{
	TEST(MatmulAutogradTest, DifferentiatesMatrixProducts)
	{
		const Tensor lhs = from_data(
			{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Shape{ 2, 3 });
		const Tensor rhs = from_data(
			{ 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F }, Shape{ 3, 2 });
		const std::array<Tensor, 2> inputs{ lhs, rhs };
		const std::vector<Tensor> gradients = grad(sum(matmul(lhs, rhs)), inputs);

		const std::array<float, 6> lhs_expected{
			15.0F, 19.0F, 23.0F,
			15.0F, 19.0F, 23.0F
		};
		const std::array<float, 6> rhs_expected{
			5.0F, 5.0F,
			7.0F, 7.0F,
			9.0F, 9.0F
		};
		expect_tensor_eq(gradients[0], lhs_expected);
		expect_tensor_eq(gradients[1], rhs_expected);
	}

	TEST(MatmulAutogradTest, RemovesSyntheticDotProductDimensions)
	{
		const std::array<float, 3> lhs_values{ 1.0F, 2.0F, 3.0F };
		const std::array<float, 3> rhs_values{ 4.0F, 5.0F, 6.0F };
		const Tensor lhs = from_data(lhs_values, Shape{ 3 });
		const Tensor rhs = from_data(rhs_values, Shape{ 3 });
		const std::array<Tensor, 2> inputs{ lhs, rhs };
		const std::vector<Tensor> gradients = grad(matmul(lhs, rhs), inputs);

		expect_tensor_eq(gradients[0], rhs_values);
		expect_tensor_eq(gradients[1], lhs_values);
	}

	TEST(MatmulAutogradTest, HandlesBatchedMatrixVectorPromotion)
	{
		const Tensor matrix = from_data(
			{
				1.0F, 2.0F, 3.0F,
				4.0F, 5.0F, 6.0F,
				-1.0F, 0.0F, 1.0F,
				2.0F, -2.0F, 3.0F
			},
			Shape{ 2, 2, 3 });
		const Tensor vector = from_data({ 2.0F, -1.0F, 3.0F }, Shape{ 3 });
		const std::array<Tensor, 2> inputs{ matrix, vector };
		const std::vector<Tensor> gradients = grad(sum(matmul(matrix, vector)), inputs);

		const std::array<float, 12> matrix_expected{
			2.0F, -1.0F, 3.0F,
			2.0F, -1.0F, 3.0F,
			2.0F, -1.0F, 3.0F,
			2.0F, -1.0F, 3.0F
		};
		const std::array<float, 3> vector_expected{ 6.0F, 5.0F, 13.0F };
		expect_tensor_eq(gradients[0], matrix_expected);
		expect_tensor_eq(gradients[1], vector_expected);
	}

	TEST(MatmulAutogradTest, HandlesBatchedVectorMatrixPromotion)
	{
		const Tensor vector = from_data({ 2.0F, -1.0F }, Shape{ 2 });
		const Tensor matrix = from_data(
			{
				1.0F, 2.0F, 3.0F,
				4.0F, 5.0F, 6.0F,
				-1.0F, 0.0F, 1.0F,
				2.0F, -2.0F, 3.0F
			},
			Shape{ 2, 2, 3 });
		const std::array<Tensor, 2> inputs{ vector, matrix };
		const std::vector<Tensor> gradients = grad(sum(matmul(vector, matrix)), inputs);

		const std::array<float, 2> vector_expected{ 6.0F, 18.0F };
		const std::array<float, 12> matrix_expected{
			2.0F, 2.0F, 2.0F,
			-1.0F, -1.0F, -1.0F,
			2.0F, 2.0F, 2.0F,
			-1.0F, -1.0F, -1.0F
		};
		expect_tensor_eq(gradients[0], vector_expected);
		expect_tensor_eq(gradients[1], matrix_expected);
	}

	TEST(MatmulAutogradTest, ReducesBroadcastBatchDimensions)
	{
		const Tensor lhs = from_data(
			{ 1.0F, 2.0F, 3.0F, 4.0F }, Shape{ 2, 1, 1, 2 });
		const Tensor rhs = from_data(
			{ 10.0F, 1.0F, 2.0F, 3.0F, -1.0F, 5.0F }, Shape{ 3, 2, 1 });
		const std::array<Tensor, 2> inputs{ lhs, rhs };
		const std::vector<Tensor> gradients = grad(sum(matmul(lhs, rhs)), inputs);

		const std::array<float, 4> lhs_expected{ 11.0F, 9.0F, 11.0F, 9.0F };
		const std::array<float, 6> rhs_expected{
			4.0F, 6.0F,
			4.0F, 6.0F,
			4.0F, 6.0F
		};
		expect_tensor_eq(gradients[0], lhs_expected);
		expect_tensor_eq(gradients[1], rhs_expected);
	}
}
