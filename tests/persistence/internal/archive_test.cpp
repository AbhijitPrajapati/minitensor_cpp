#include <minitensor/persistence/detail/archive.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>

namespace minitensor::persistence::test
{
namespace
{
// Independently specified version-1 fixture: scalar Float32 named "x", value 2.5.
// Descriptor starts at 28, name length at 33, index at 42, padding at 50, payload at 56.
std::string scalar_fixture()
{
    constexpr char bytes[] = "MTARCH\r\n"
                             "\x01\x00\x00\x00"                 // Version.
                             "\x01\x00\x00\x00\x00\x00\x00\x00" // Tensor count.
                             "\x01\x00\x00\x00\x00\x00\x00\x00" // Binding count.
                             "\x01"                             // Float32 tag.
                             "\x00\x00\x00\x00"                 // Scalar rank.
                             "\x01\x00\x00\x00\x00\x00\x00\x00" // Name length.
                             "x"
                             "\x00\x00\x00\x00\x00\x00\x00\x00" // Tensor index.
                             "\x00\x00\x00\x00\x00\x00"         // Padding.
                             "\x00\x00\x20\x40";                // IEEE 754 binary32, little-endian.
    return {bytes, sizeof(bytes) - 1};
}

detail::ArchiveData decode(const std::string &bytes)
{
    std::istringstream input(bytes);
    return detail::read_archive(input, Device::cpu());
}

class NonSeekingBuffer final : public std::stringbuf
{
  public:
    explicit NonSeekingBuffer(const std::string &bytes) : std::stringbuf(bytes)
    {
    }

  protected:
    pos_type seekoff(off_type, std::ios_base::seekdir, std::ios_base::openmode) override
    {
        throw std::logic_error{"unexpected seek"};
    }
    pos_type seekpos(pos_type, std::ios_base::openmode) override
    {
        throw std::logic_error{"unexpected seek"};
    }
};

class ShortWriteBuffer final : public std::streambuf
{
  protected:
    std::streamsize xsputn(const char *, std::streamsize count) override
    {
        return count - 1;
    }
};
} // namespace

TEST(ArchiveTest, ReadsAndWritesKnownLittleEndianFixtures)
{
    auto bytes = scalar_fixture();
    const auto floats = decode(bytes);
    ASSERT_EQ(floats.tensors.size(), 1U);
    EXPECT_EQ(floats.bindings.at("x"), 0U);
    EXPECT_EQ(floats.tensors[0].shape(), Shape{});
    EXPECT_EQ(floats.tensors[0].dtype(), DType::Float32);
    EXPECT_FLOAT_EQ(item(floats.tensors[0]), 2.5F);
    std::ostringstream float_output;
    detail::write_archive(float_output, floats);
    EXPECT_EQ(float_output.str(), bytes);

    bytes[28] = '\x02'; // Int64 tag, with an independently specified 64-bit payload.
    bytes.replace(56, 4, "\xef\xcd\xab\x89\x67\x45\x23\x01", 8);
    const auto integers = decode(bytes);
    ASSERT_EQ(integers.tensors.size(), 1U);
    EXPECT_EQ(integers.tensors[0].dtype(), DType::Int64);
    EXPECT_EQ(item<std::int64_t>(integers.tensors[0]), INT64_C(0x0123456789abcdef));
    std::ostringstream integer_output;
    detail::write_archive(integer_output, integers);
    EXPECT_EQ(integer_output.str(), bytes);
}

TEST(ArchiveTest, AllowsAliasesAndUnnamedTensorRecords)
{
    const Tensor value = full(Shape{}, 3.0F);
    std::stringstream stream;
    detail::write_archive(stream, {{value, value}, {{"first", 0}, {"alias", 0}}});
    const auto data = detail::read_archive(stream, Device::cpu());
    ASSERT_EQ(data.tensors.size(), 2U);
    EXPECT_EQ(data.bindings.at("first"), data.bindings.at("alias"));
    EXPECT_FLOAT_EQ(item(data.tensors[1]), 3.0F);
}

TEST(ArchiveTest, SupportsNonSeekingStreamsAndEnabledStreamExceptions)
{
    NonSeekingBuffer buffer(scalar_fixture());
    std::istream input(&buffer);
    EXPECT_FLOAT_EQ(item(detail::read_archive(input, Device::cpu()).tensors.at(0)), 2.5F);

    std::istringstream exceptions(scalar_fixture());
    exceptions.exceptions(std::ios::badbit | std::ios::failbit | std::ios::eofbit);
    EXPECT_NO_THROW((void)detail::read_archive(exceptions, Device::cpu()));
}

TEST(ArchiveTest, RejectsUnknownMagicVersionsAndDtypeTags)
{
    for (const std::size_t offset : {0U, 8U, 28U})
    {
        SCOPED_TRACE(offset);
        auto bytes = scalar_fixture();
        bytes[offset] = '\xff';
        EXPECT_THROW((void)decode(bytes), std::invalid_argument);
    }
}

TEST(ArchiveTest, RejectsExcessiveCountsAndStringLengthsBeforeAllocation)
{
    for (const std::size_t offset : {12U, 20U, 33U})
    {
        SCOPED_TRACE(offset);
        auto bytes = scalar_fixture();
        bytes.replace(offset, 8, 8, '\xff');
        EXPECT_THROW((void)decode(bytes), std::length_error);
    }
}

TEST(ArchiveTest, RejectsExcessiveRanksAndUnrepresentableExtents)
{
    auto bytes = scalar_fixture();
    bytes.replace(29, 4, "\x01\x08\x00\x00", 4); // Rank 2049.
    EXPECT_THROW((void)decode(bytes), std::length_error);

    bytes = scalar_fixture().substr(0, 33);
    bytes[29] = '\x01';
    bytes.append(8, '\xff');
    EXPECT_THROW((void)decode(bytes), std::overflow_error);
}

TEST(ArchiveTest, RejectsElementCountAndByteCountOverflow)
{
    auto bytes = scalar_fixture().substr(0, 33);
    bytes[29] = '\x01';
    bytes.append("\xff\xff\xff\xff\xff\xff\xff\x7f", 8);
    EXPECT_THROW((void)decode(bytes), std::overflow_error);

    bytes[29] = '\x02';
    bytes.append("\x03\x00\x00\x00\x00\x00\x00\x00", 8);
    EXPECT_THROW((void)decode(bytes), std::overflow_error);
}

TEST(ArchiveTest, RejectsExcessivePayloadsBeforeAllocationOrEvaluation)
{
    auto bytes = scalar_fixture().substr(0, 33);
    bytes[29] = '\x01';
    bytes.append("\x01\x00\x00\x10\x00\x00\x00\x00", 8); // 2^28 + 1 Float32 elements.
    EXPECT_THROW((void)decode(bytes), std::length_error);

    // Each lazy tensor fits separately, but together they exceed the archive limit.
    const Tensor large = full(Shape{150'000'000}, 1.0F);
    std::ostringstream output;
    EXPECT_THROW(detail::write_archive(output, {{large, large}, {}}), std::length_error);
    EXPECT_TRUE(output.str().empty());
}

TEST(ArchiveTest, RejectsInvalidBindingIndicesOnBothSides)
{
    auto bytes = scalar_fixture();
    bytes[42] = '\x01';
    EXPECT_THROW((void)decode(bytes), std::invalid_argument);

    std::ostringstream output;
    EXPECT_THROW(detail::write_archive(output, {{full(Shape{}, 1.0F)}, {{"x", 1}}}), std::invalid_argument);
    EXPECT_TRUE(output.str().empty());
}

TEST(ArchiveTest, RejectsDuplicateBindingNames)
{
    const auto fixture = scalar_fixture();
    auto bytes = fixture.substr(0, 33);
    bytes[20] = '\x02';
    bytes += fixture.substr(33, 17);
    bytes += fixture.substr(33, 17);
    bytes.append(5, '\0');
    bytes += fixture.substr(56);
    EXPECT_THROW((void)decode(bytes), std::invalid_argument);
}

TEST(ArchiveTest, RejectsNonzeroPaddingAndTrailingBytes)
{
    auto bytes = scalar_fixture();
    bytes[50] = '\x01';
    EXPECT_THROW((void)decode(bytes), std::invalid_argument);
    EXPECT_THROW((void)decode(scalar_fixture() + '\0'), std::invalid_argument);
}

TEST(ArchiveTest, RejectsTruncationAtEveryBoundary)
{
    const auto bytes = scalar_fixture();
    for (std::size_t length = 0; length < bytes.size(); ++length)
    {
        SCOPED_TRACE(length);
        EXPECT_THROW((void)decode(bytes.substr(0, length)), std::runtime_error);
    }
}

TEST(ArchiveTest, EnforcesNameAndRankLimitsWhenWriting)
{
    const Tensor scalar = full(Shape{}, 1.0F);
    std::ostringstream names;
    EXPECT_THROW(detail::write_archive(names, {{scalar}, {{std::string((1 << 20) + 1, 'x'), 0}}}), std::length_error);
    EXPECT_TRUE(names.str().empty());

    const Tensor high_rank = full(Shape{std::vector<Extent>(2049, 1)}, 1.0F);
    std::ostringstream ranks;
    EXPECT_THROW(detail::write_archive(ranks, {{high_rank}, {}}), std::length_error);
    EXPECT_TRUE(ranks.str().empty());
}

TEST(ArchiveTest, ReportsFailedStreamsAndShortWrites)
{
    std::istringstream input(scalar_fixture());
    input.setstate(std::ios::badbit);
    EXPECT_THROW((void)detail::read_archive(input, Device::cpu()), std::runtime_error);

    std::ostringstream output;
    output.setstate(std::ios::badbit);
    EXPECT_THROW(detail::write_archive(output, {}), std::runtime_error);

    ShortWriteBuffer buffer;
    std::ostream short_output(&buffer);
    EXPECT_THROW(detail::write_archive(short_output, {}), std::runtime_error);
}
} // namespace minitensor::persistence::test
