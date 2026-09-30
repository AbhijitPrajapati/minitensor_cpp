#include <minitensor/nn/parameter.hpp>

#include <functional>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::nn::test
{
	static_assert(!std::is_default_constructible_v<ParameterId>);
	static_assert(!std::is_constructible_v<ParameterId, ParameterId::value_type>);

	TEST(ParameterIdTest, IsOpaqueComparableAndHashable)
	{
		const Parameter first{ full(Shape{ 2 }, 1.0F) };
		const Parameter copied = first;
		const Parameter second{ full(Shape{ 2 }, 1.0F) };

		EXPECT_EQ(copied.id(), first.id());
		EXPECT_NE(second.id(), first.id());
		EXPECT_EQ(std::hash<ParameterId>{}(copied.id()),
			std::hash<ParameterId>{}(first.id()));

		const std::unordered_set<ParameterId> ids{
			first.id(), copied.id(), second.id()
		};
		EXPECT_EQ(ids.size(), 2);
		EXPECT_TRUE(ids.contains(first.id()));
		EXPECT_TRUE(ids.contains(second.id()));
	}

	TEST(ParameterTest, ExposesItsValueAndConstructionMetadata)
	{
		const Parameter defaults{ full(Shape{ 2, 1 }, 3.0F) };
		EXPECT_TRUE(defaults.metadata().trainable);
		EXPECT_EQ(defaults.value().shape(), (Shape{ 2, 1 }));
		EXPECT_EQ(defaults.value().dtype(), DType::Float32);
		EXPECT_EQ(defaults.value().device(), Device::cpu());
		EXPECT_EQ(to_vector(defaults.value()),
			(std::vector<float>{ 3.0F, 3.0F }));

		const Parameter frozen{
			full(Shape{}, 4.0F), ParameterMetadata{ .trainable = false }
		};
		EXPECT_FALSE(frozen.metadata().trainable);
		EXPECT_FLOAT_EQ(item(frozen.value()), 4.0F);
	}

	TEST(ParameterTest, CopyAndMovePreserveIdentityAndState)
	{
		const Parameter original{
			full(Shape{ 2 }, 2.0F), ParameterMetadata{ .trainable = false }
		};
		const ParameterId id = original.id();

		Parameter copied = original;
		EXPECT_EQ(copied.id(), id);
		EXPECT_FALSE(copied.metadata().trainable);
		EXPECT_EQ(to_vector(copied.value()),
			(std::vector<float>{ 2.0F, 2.0F }));

		Parameter moved = std::move(copied);
		EXPECT_EQ(moved.id(), id);
		EXPECT_FALSE(moved.metadata().trainable);
		EXPECT_EQ(to_vector(moved.value()),
			(std::vector<float>{ 2.0F, 2.0F }));

		Parameter copy_assigned{ full(Shape{}, 0.0F) };
		copy_assigned = original;
		EXPECT_EQ(copy_assigned.id(), id);

		Parameter move_assigned{ full(Shape{}, 0.0F) };
		move_assigned = std::move(copy_assigned);
		EXPECT_EQ(move_assigned.id(), id);
		EXPECT_FALSE(move_assigned.metadata().trainable);
	}

	TEST(ParameterTest, WithValuePreservesIdentityAndMetadata)
	{
		const Parameter original{
			full(Shape{ 2 }, 1.0F), ParameterMetadata{ .trainable = false }
		};
		const Parameter updated = original.with_value(full(Shape{ 2 }, 7.0F));

		EXPECT_EQ(updated.id(), original.id());
		EXPECT_EQ(updated.metadata(), original.metadata());
		EXPECT_EQ(to_vector(updated.value()),
			(std::vector<float>{ 7.0F, 7.0F }));
		EXPECT_EQ(to_vector(original.value()),
			(std::vector<float>{ 1.0F, 1.0F }));
	}

	TEST(ParameterTest, WithValueRejectsIncompatibleSpecifications)
	{
		const Parameter parameter{ full(Shape{ 2 }, 1.0F) };

		EXPECT_THROW(
			(void)parameter.with_value(full(Shape{ 2, 1 }, 1.0F)),
			std::invalid_argument);
		EXPECT_THROW(
			(void)parameter.with_value(full(
				Shape{ 2 }, 1.0F,
				TensorOptions{ DType::Float32, Device::cpu(1) })),
			std::invalid_argument);
	}

	TEST(ParameterTest, WithMetadataPreservesIdentityAndValue)
	{
		const Parameter original{ full(Shape{ 2 }, 6.0F) };
		const Parameter updated = original.with_metadata(
			ParameterMetadata{ .trainable = false });

		EXPECT_EQ(updated.id(), original.id());
		EXPECT_FALSE(updated.metadata().trainable);
		EXPECT_TRUE(original.metadata().trainable);
		EXPECT_EQ(to_vector(updated.value()),
			(std::vector<float>{ 6.0F, 6.0F }));
	}
}
