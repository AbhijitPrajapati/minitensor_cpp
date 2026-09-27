#include <minitensor/autograd.hpp>
#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/manipulation.hpp>
#include <minitensor/ops/reduction.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <array>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/support/tensor_assertions.hpp"

namespace minitensor::test
{
	TEST(ManipulationAutogradTest, InvertsReshapeAndPermutation)
	{
		const Tensor input = from_data(
			{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Shape{ 2, 3 });
		const Tensor output = permute(reshape(input, Shape{ 3, 2 }), { 1, 0 });
		const Tensor seed = from_data(
			{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Shape{ 2, 3 });
		const Tensor result = vjp(output, input, seed).front();
		const std::array<float, 6> expected{ 1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F };

		EXPECT_EQ(result.shape(), input.shape());
		expect_tensor_eq(result, expected);
	}

	TEST(ManipulationAutogradTest, PropagatesThroughContiguousCopies)
	{
		const Tensor input = from_data(
			{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Shape{ 2, 3 });
		const Tensor gradient = grad(sum(contiguous(transpose(input))), input).front();
		const std::array<float, 6> expected{ 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F };
		expect_tensor_eq(gradient, expected);
	}

	TEST(ManipulationAutogradTest, ReducesBroadcastCotangents)
	{
		const Tensor input = from_data({ 10.0F, 20.0F, 30.0F }, Shape{ 1, 3 });
		const Tensor output = broadcast_to(input, Shape{ 2, 3 });
		const Tensor seed = from_data(
			{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Shape{ 2, 3 });
		const Tensor result = vjp(output, input, seed).front();
		const std::array<float, 3> expected{ 5.0F, 7.0F, 9.0F };

		EXPECT_EQ(result.shape(), input.shape());
		expect_tensor_eq(result, expected);
	}

	TEST(ManipulationAutogradTest, ScattersSliceCotangents)
	{
		const Tensor input = from_data(
			{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Shape{ 2, 3 });
		const Tensor output = slice(input, -1, std::nullopt, std::nullopt, -1);
		const Tensor seed = from_data(
			{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Shape{ 2, 3 });
		const Tensor result = vjp(output, input, seed).front();
		const std::array<float, 6> expected{ 3.0F, 2.0F, 1.0F, 6.0F, 5.0F, 4.0F };
		expect_tensor_eq(result, expected);
	}

	TEST(ManipulationAutogradTest, ZeroFillsAnEmptySliceCotangent)
	{
		const Tensor input = full(Shape{ 2, 3 }, 1.0F);
		const Tensor output = slice(input, 1, 1, 1);
		const Tensor seed = full(Shape{ 2, 0 }, 1.0F);
		const Tensor result = vjp(output, input, seed).front();
		const std::array<float, 6> expected{};
		expect_tensor_eq(result, expected);
	}

	TEST(ManipulationAutogradTest, SplitsConcatenationCotangents)
	{
		const Tensor lhs = full(Shape{ 2, 1 }, 0.0F);
		const Tensor rhs = full(Shape{ 2, 2 }, 0.0F);
		const std::array<Tensor, 2> inputs{ lhs, rhs };
		const Tensor output = concatenate(inputs, 1);
		const Tensor seed = from_data(
			{ 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F }, Shape{ 2, 3 });
		const std::vector<Tensor> results = vjp(output, inputs, seed);

		ASSERT_EQ(results.size(), 2);
		const std::array<float, 2> lhs_expected{ 1.0F, 4.0F };
		const std::array<float, 4> rhs_expected{ 2.0F, 3.0F, 5.0F, 6.0F };
		expect_tensor_eq(results[0], lhs_expected);
		expect_tensor_eq(results[1], rhs_expected);
	}

	TEST(ManipulationAutogradTest, BroadcastsReductionCotangents)
	{
		const Tensor input = full(Shape{ 2, 3 }, 0.0F);
		const Tensor output = sum(input, { 1 }, true);
		const Tensor seed = from_data({ 2.0F, 4.0F }, Shape{ 2, 1 });
		const Tensor result = vjp(output, input, seed).front();
		const std::array<float, 6> expected{ 2.0F, 2.0F, 2.0F, 4.0F, 4.0F, 4.0F };
		expect_tensor_eq(result, expected);
	}
}
