#include <array>
#include <optional>
#include <stdexcept>

#include <gtest/gtest.h>

#include <minitensor/ops/creation.hpp>

#include "tensor/autograd/vjp_registry.hpp"
#include "tensor/autograd/vjp_registry.hpp"
#include "tensor/primitives/elementwise/add.hpp"
#include "tensor/primitives/elementwise/sign.hpp"
#include "tensor/primitives/linalg/matmul.hpp"
#include "tensor/primitives/manipulation/basic_index.hpp"
#include "tensor/primitives/manipulation/gather.hpp"
#include "tensor/primitives/reduction/logsumexp.hpp"
#include "tensor/support/test_primitive.hpp"

namespace minitensor::test
{
	TEST(VjpRegistryTest, RegistersAndInvokesTypedRule)
	{
		detail::VjpRegistry registry;
		registry.register_rule<IdentitySpecPrimitive>(
			[](const IdentitySpecPrimitive&, const detail::VjpContext& context)
			{
				return detail::VjpContributions{ context.output_cotangent };
			});

		const Tensor input = full(Shape{ 2 }, 1.0F);
		const Tensor output = full(Shape{ 2 }, 2.0F);
		const Tensor cotangent = full(Shape{ 2 }, 3.0F);
		const std::array<Tensor, 1> inputs{ input };
		const detail::VjpContext context{ inputs, output, cotangent };
		const detail::VjpContributions result =
			registry.get(IdentitySpecPrimitive::ID)(IdentitySpecPrimitive{}, context);

		ASSERT_EQ(result.size(), 1);
		ASSERT_TRUE(result.front().has_value());
		EXPECT_EQ(result.front()->shape(), cotangent.shape());
	}

	TEST(VjpRegistryTest, RejectsInvalidRegistrationAndLookup)
	{
		detail::VjpRegistry registry;
		EXPECT_THROW(
			registry.register_rule<IdentitySpecPrimitive>(
				detail::TypedVjpRule<IdentitySpecPrimitive>{}),
			std::invalid_argument);

		registry.register_rule<IdentitySpecPrimitive>(
			[](const IdentitySpecPrimitive&, const detail::VjpContext&)
			{
				return detail::VjpContributions{};
			});
		EXPECT_THROW(
			registry.register_rule<IdentitySpecPrimitive>(
				[](const IdentitySpecPrimitive&, const detail::VjpContext&)
				{
					return detail::VjpContributions{};
				}),
			std::logic_error);
		EXPECT_THROW(
			(void)registry.get(DestructionTrackedPrimitive::ID),
			std::runtime_error);
	}

	TEST(VjpRegistryTest, BuiltInRegistryContainsDifferentiablePrimitiveFamilies)
	{
		const detail::VjpRegistry& registry = detail::vjp_registry();
		EXPECT_TRUE(registry.contains(detail::AddPrimitive::ID));
		EXPECT_TRUE(registry.contains(detail::SignPrimitive::ID));
		EXPECT_TRUE(registry.contains(detail::MatmulPrimitive::ID));
		EXPECT_TRUE(registry.contains(detail::BasicIndexPrimitive::ID));
		EXPECT_TRUE(registry.contains(detail::BasicIndexScatterPrimitive::ID));
		EXPECT_TRUE(registry.contains(detail::GatherPrimitive::ID));
		EXPECT_TRUE(registry.contains(detail::GatherScatterAddPrimitive::ID));
		EXPECT_TRUE(registry.contains(detail::LogSumExpPrimitive::ID));
	}
}
