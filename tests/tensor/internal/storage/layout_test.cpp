#include <minitensor/types.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/storage/layout.hpp"

namespace minitensor::test
{
	TEST(LayoutTest, RepresentsScalarAndCustomLayouts)
	{
		const detail::Layout scalar;
		EXPECT_EQ(scalar.rank(), 0);
		EXPECT_TRUE(scalar.strides().empty());
		EXPECT_EQ(scalar.offset(), 0);
		EXPECT_TRUE(scalar.is_contiguous(Shape{}));

		const detail::Layout custom{ { 12, 4, 1 }, 5 };
		const std::array<detail::Layout::stride_type, 3> expected{ 12, 4, 1 };
		EXPECT_EQ(custom.rank(), expected.size());
		EXPECT_TRUE(std::ranges::equal(custom.strides(), expected));
		EXPECT_EQ(custom.stride(1), 4);
		EXPECT_EQ(custom.offset(), 5);
		EXPECT_EQ(custom, (detail::Layout{ { 12, 4, 1 }, 5 }));
	}

	TEST(LayoutTest, RejectsNegativeBaseOffsets)
	{
		EXPECT_THROW(
			(void)(detail::Layout{ { 1 }, -1 }),
			std::invalid_argument);
	}

	TEST(LayoutTest, ConstructsContiguousStrides)
	{
		const detail::Layout layout = detail::Layout::contiguous(Shape{ 2, 3, 4 });
		EXPECT_EQ(layout, (detail::Layout{ { 12, 4, 1 } }));
		EXPECT_TRUE(layout.is_contiguous(Shape{ 2, 3, 4 }));
		EXPECT_EQ(
			detail::Layout::contiguous(Shape{ 2, 3, 4 }, 6),
			(detail::Layout{ { 12, 4, 1 }, 6 }));
	}

	TEST(LayoutTest, DetectsContiguitySemantically)
	{
		EXPECT_TRUE((detail::Layout{ { 3, 99, 1 }, 7 }).is_contiguous(
			Shape{ 2, 1, 3 }));
		EXPECT_FALSE((detail::Layout{ { 1, 2 } }).is_contiguous(Shape{ 2, 3 }));
		EXPECT_FALSE((detail::Layout{ { 3, 1 } }).is_contiguous(Shape{ 2, 1, 3 }));
		EXPECT_TRUE((detail::Layout{ { 123, -45 } }).is_contiguous(Shape{ 4, 0 }));
	}

	TEST(LayoutTest, PermutesStridesAndPreservesOffset)
	{
		const detail::Layout layout{ { 12, 4, 1 }, 5 };
		const std::array<Shape::size_type, 3> permutation{ 2, 0, 1 };
		EXPECT_EQ(
			layout.permuted(permutation),
			(detail::Layout{ { 1, 12, 4 }, 5 }));
	}

	TEST(LayoutTest, ReshapesOnlyStorageCompatibleLayouts)
	{
		const detail::Layout contiguous{ { 12, 4, 1 }, 5 };
		const auto reshaped = contiguous.try_reshape(Shape{ 2, 3, 4 }, Shape{ 4, 6 });
		ASSERT_TRUE(reshaped.has_value());
		EXPECT_EQ(*reshaped, (detail::Layout{ { 6, 1 }, 5 }));

		const detail::Layout noncontiguous{ { 1, 2 }, 4 };
		const auto unchanged = noncontiguous.try_reshape(Shape{ 2, 3 }, Shape{ 2, 3 });
		ASSERT_TRUE(unchanged.has_value());
		EXPECT_EQ(*unchanged, noncontiguous);
		EXPECT_FALSE(noncontiguous.try_reshape(Shape{ 2, 3 }, Shape{ 3, 2 }));
		EXPECT_FALSE(contiguous.try_reshape(Shape{ 2, 3, 4 }, Shape{ 5, 5 }));

		const detail::Layout scalar{ std::vector<detail::Layout::stride_type>{}, 7 };
		const auto singleton = scalar.try_reshape(Shape{}, Shape{ 1, 1 });
		ASSERT_TRUE(singleton.has_value());
		EXPECT_EQ(*singleton, (detail::Layout{ { 1, 1 }, 7 }));
	}

	TEST(LayoutTest, BroadcastsStrides)
	{
		const detail::Layout source{ { 3, 3, 1 }, 5 };
		EXPECT_EQ(
			source.broadcasted_to(Shape{ 2, 1, 3 }, Shape{ 2, 4, 3 }),
			(detail::Layout{ { 3, 0, 1 }, 5 }));
		EXPECT_EQ(
			(detail::Layout{ { 1 }, 2 }).broadcasted_to(Shape{ 3 }, Shape{ 4, 3 }),
			(detail::Layout{ { 0, 1 }, 2 }));
		EXPECT_EQ(
			detail::Layout{}.broadcasted_to(Shape{}, Shape{ 2, 3 }),
			(detail::Layout{ { 0, 0 } }));
	}

	TEST(LayoutTest, ValidatesBroadcasting)
	{
		const detail::Layout rank_one{ { 1 } };
		const detail::Layout rank_two{ { 3, 1 } };
		EXPECT_THROW(
			(void)rank_one.broadcasted_to(Shape{ 2, 3 }, Shape{ 2, 3 }),
			std::invalid_argument);
		EXPECT_THROW(
			(void)rank_two.broadcasted_to(Shape{ 2, 3 }, Shape{ 3 }),
			std::invalid_argument);
		EXPECT_THROW(
			(void)rank_two.broadcasted_to(Shape{ 2, 3 }, Shape{ 2, 4 }),
			std::invalid_argument);
	}

	TEST(LayoutTest, RejectsContiguousStrideOverflow)
	{
		const Shape shape{ 0, std::numeric_limits<Extent>::max(), 2 };
		EXPECT_THROW((void)detail::Layout::contiguous(shape), std::overflow_error);
	}
}
