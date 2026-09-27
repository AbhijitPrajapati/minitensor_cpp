#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/autograd/engine.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/tensor_access.hpp"
#include "tensor/support/test_primitive.hpp"
#include "tensor/support/tensor_assertions.hpp"

namespace minitensor::test
{
	namespace
	{
		Tensor apply_vjp_rule(
			std::span<const Tensor> inputs,
			detail::TensorSpec output_spec,
			RuleBasedVjpPrimitive::Rule rule)
		{
			std::vector<detail::ValueRef> values;
			values.reserve(inputs.size());
			for (const Tensor& input : inputs)
			{
				values.push_back(detail::TensorAccess::value(input));
			}
			return detail::TensorAccess::make(detail::apply_operation(
				std::make_unique<RuleBasedVjpPrimitive>(
					inputs.size(), std::move(output_spec), std::move(rule)),
				values));
		}
	}

	TEST(ReverseVjpTest, InvokesRulesInReverseTopologicalOrder)
	{
		const detail::TensorSpec spec{ Shape{ 2 }, DType::Float32, Device::cpu() };
		const Tensor source = full(spec.shape, 3.0F);
		const Tensor seed = from_data({ 1.5F, -2.0F }, spec.shape);
		std::vector<int> calls;
		const detail::Value* first_value = nullptr;

		const std::array<Tensor, 1> first_inputs{ source };
		const Tensor first = apply_vjp_rule(
			first_inputs,
			spec,
			[&](std::span<const Tensor> inputs, const Tensor& output, const Tensor& cotangent)
			{
				calls.push_back(1);
				EXPECT_EQ(inputs.size(), 1);
				EXPECT_EQ(detail::TensorAccess::value(inputs.front()), detail::TensorAccess::value(source));
				EXPECT_EQ(detail::TensorAccess::value(output).get(), first_value);
				EXPECT_EQ(detail::TensorAccess::value(cotangent), detail::TensorAccess::value(seed));
				return std::vector<std::optional<Tensor>>{ cotangent };
			});
		first_value = detail::TensorAccess::value(first).get();

		const detail::Value* output_value = nullptr;
		const std::array<Tensor, 1> output_inputs{ first };
		const Tensor output = apply_vjp_rule(
			output_inputs,
			spec,
			[&](std::span<const Tensor> inputs, const Tensor& node_output, const Tensor& cotangent)
			{
				calls.push_back(2);
				EXPECT_EQ(detail::TensorAccess::value(inputs.front()), detail::TensorAccess::value(first));
				EXPECT_EQ(detail::TensorAccess::value(node_output).get(), output_value);
				EXPECT_EQ(detail::TensorAccess::value(cotangent), detail::TensorAccess::value(seed));
				return std::vector<std::optional<Tensor>>{ cotangent };
			});
		output_value = detail::TensorAccess::value(output).get();

		const std::array<detail::ValueRef, 1> targets{ detail::TensorAccess::value(source) };
		const std::vector<Tensor> result = detail::reverse_vjp(
			detail::TensorAccess::value(output), targets, seed);

		EXPECT_EQ(calls, (std::vector<int>{ 2, 1 }));
		ASSERT_EQ(result.size(), 1);
		EXPECT_EQ(detail::TensorAccess::value(result.front()), detail::TensorAccess::value(seed));
	}

	TEST(ReverseVjpTest, SkipsTraversalWhenNoTargetsAreRequested)
	{
		const detail::TensorSpec spec{ Shape{ 2 }, DType::Float32, Device::cpu() };
		const Tensor source = full(spec.shape, 1.0F);
		std::size_t calls = 0;
		const std::array<Tensor, 1> inputs{ source };
		const Tensor output = apply_vjp_rule(
			inputs,
			spec,
			[&calls](std::span<const Tensor>, const Tensor&, const Tensor& cotangent)
			{
				++calls;
				return std::vector<std::optional<Tensor>>{ cotangent };
			});
		const Tensor seed = full(spec.shape, 1.0F);

		const std::vector<Tensor> result = detail::reverse_vjp(
			detail::TensorAccess::value(output),
			std::span<const detail::ValueRef>{},
			seed);
		EXPECT_TRUE(result.empty());
		EXPECT_EQ(calls, 0);
	}

	TEST(ReverseVjpTest, AccumulatesDuplicateInputContributions)
	{
		const detail::TensorSpec spec{ Shape{ 2 }, DType::Float32, Device::cpu() };
		const Tensor source = full(spec.shape, 1.0F);
		const std::array<Tensor, 2> inputs{ source, source };
		const Tensor output = apply_vjp_rule(
			inputs,
			spec,
			[](std::span<const Tensor>, const Tensor&, const Tensor& cotangent)
			{
				return std::vector<std::optional<Tensor>>{ cotangent, cotangent };
			});
		const Tensor seed = from_data({ 1.5F, -2.0F }, spec.shape);
		const std::array<detail::ValueRef, 1> targets{ detail::TensorAccess::value(source) };

		const Tensor result = detail::reverse_vjp(
			detail::TensorAccess::value(output), targets, seed).front();
		const std::array<float, 2> expected{ 3.0F, -4.0F };
		expect_tensor_eq(result, expected);
	}

	TEST(ReverseVjpTest, PrunesBranchesThatCannotReachTargets)
	{
		const detail::TensorSpec spec{ Shape{ 2 }, DType::Float32, Device::cpu() };
		const Tensor source = full(spec.shape, 1.0F);
		const Tensor unrelated = full(spec.shape, 2.0F);
		std::size_t unrelated_calls = 0;
		const std::array<Tensor, 1> unrelated_inputs{ unrelated };
		const Tensor unrelated_branch = apply_vjp_rule(
			unrelated_inputs,
			spec,
			[&unrelated_calls](std::span<const Tensor>, const Tensor&, const Tensor& cotangent)
			{
				++unrelated_calls;
				return std::vector<std::optional<Tensor>>{ cotangent };
			});

		std::size_t join_calls = 0;
		const std::array<Tensor, 2> join_inputs{ source, unrelated_branch };
		const Tensor output = apply_vjp_rule(
			join_inputs,
			spec,
			[&join_calls](std::span<const Tensor>, const Tensor&, const Tensor& cotangent)
			{
				++join_calls;
				return std::vector<std::optional<Tensor>>{ cotangent, cotangent };
			});
		const std::array<detail::ValueRef, 1> targets{ detail::TensorAccess::value(source) };
		const Tensor seed = full(spec.shape, 1.0F);

		const std::vector<Tensor> result = detail::reverse_vjp(
			detail::TensorAccess::value(output), targets, seed);
		EXPECT_EQ(result.size(), 1);
		EXPECT_EQ(join_calls, 1);
		EXPECT_EQ(unrelated_calls, 0);
	}

	TEST(ReverseVjpTest, TreatsMissingContributionsAsZero)
	{
		const detail::TensorSpec spec{ Shape{ 2 }, DType::Float32, Device::cpu() };
		const Tensor source = full(spec.shape, 1.0F);
		const std::array<Tensor, 1> inputs{ source };
		const Tensor output = apply_vjp_rule(
			inputs,
			spec,
			[](std::span<const Tensor>, const Tensor&, const Tensor&)
			{
				return std::vector<std::optional<Tensor>>{ std::nullopt };
			});
		const std::array<detail::ValueRef, 1> targets{ detail::TensorAccess::value(source) };

		const Tensor result = detail::reverse_vjp(
			detail::TensorAccess::value(output),
			targets,
			full(spec.shape, 1.0F)).front();
		const std::array<float, 2> expected{};
		expect_tensor_eq(result, expected);
	}

	TEST(ReverseVjpTest, ValidatesOutputAndTargets)
	{
		const Tensor seed = full(Shape{ 2 }, 1.0F);
		EXPECT_THROW(
			(void)detail::reverse_vjp(
				detail::ValueRef{}, std::span<const detail::ValueRef>{}, seed),
			std::invalid_argument);

		const Tensor output = full(Shape{ 2 }, 1.0F);
		const std::array<detail::ValueRef, 1> null_targets{ detail::ValueRef{} };
		EXPECT_THROW(
			(void)detail::reverse_vjp(
				detail::TensorAccess::value(output), null_targets, seed),
			std::invalid_argument);
	}

	TEST(ReverseVjpTest, RejectsMalformedPrimitiveRules)
	{
		const detail::TensorSpec spec{ Shape{ 2 }, DType::Float32, Device::cpu() };
		const Tensor source = full(spec.shape, 1.0F);
		const std::array<Tensor, 1> inputs{ source };
		const std::array<detail::ValueRef, 1> targets{ detail::TensorAccess::value(source) };
		const Tensor seed = full(spec.shape, 1.0F);

		const Tensor wrong_count = apply_vjp_rule(
			inputs,
			spec,
			[](std::span<const Tensor>, const Tensor&, const Tensor&)
			{
				return std::vector<std::optional<Tensor>>{};
			});
		EXPECT_THROW(
			(void)detail::reverse_vjp(
				detail::TensorAccess::value(wrong_count), targets, seed),
			std::logic_error);

		const Tensor wrong_spec = apply_vjp_rule(
			inputs,
			spec,
			[](std::span<const Tensor>, const Tensor&, const Tensor&)
			{
				return std::vector<std::optional<Tensor>>{ full(Shape{ 1 }, 1.0F) };
			});
		EXPECT_THROW(
			(void)detail::reverse_vjp(
				detail::TensorAccess::value(wrong_spec), targets, seed),
			std::logic_error);

		const std::array<detail::ValueRef, 1> values{ detail::TensorAccess::value(source) };
		const detail::ValueRef no_rule = detail::apply_operation(
			std::make_unique<IdentitySpecPrimitive>(), values);
		EXPECT_THROW(
			(void)detail::reverse_vjp(no_rule, targets, seed),
			std::logic_error);
	}
}
