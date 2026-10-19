#include <minitensor/host_scalar.hpp>
#include <minitensor/types.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

namespace minitensor::test
{
static_assert(TensorElement<float>);
static_assert(TensorElement<std::int64_t>);
static_assert(!TensorElement<int>);

TEST(HostScalarTest, CanonicalizesCppNumericTypes)
{
    const HostScalar floating{1.25};
    EXPECT_EQ(floating.dtype(), DType::Float32);
    EXPECT_FLOAT_EQ(floating.get<float>(), 1.25F);

    const HostScalar integral{7};
    EXPECT_EQ(integral.dtype(), DType::Int64);
    EXPECT_EQ(integral.get<std::int64_t>(), 7);
}

TEST(HostScalarTest, CastsToCanonicalTensorElementTypes)
{
    const HostScalar integral = HostScalar{3.75F}.cast(DType::Int64);
    EXPECT_EQ(integral, HostScalar{std::int64_t{3}});

    const HostScalar floating = HostScalar{std::int64_t{16'777'217}}.cast(DType::Float32);
    EXPECT_EQ(floating, HostScalar{static_cast<float>(16'777'217)});
}

TEST(HostScalarTest, RejectsInvalidConversions)
{
    EXPECT_THROW((void)HostScalar{std::numeric_limits<double>::max()}, std::out_of_range);
    EXPECT_THROW((void)HostScalar{std::numeric_limits<float>::infinity()}.cast(DType::Int64), std::invalid_argument);
    EXPECT_THROW((void)HostScalar{1.0F}.get<std::int64_t>(), std::logic_error);
}
} // namespace minitensor::test
