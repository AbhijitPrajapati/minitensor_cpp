#include <minitensor/persistence/tensors.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/manipulation.hpp>
#include <minitensor/persistence/detail/archive.hpp>

namespace minitensor::persistence::test
{
TEST(TensorPersistenceTest, RoundTripsNamesShapesAndBothDtypes)
{
    const std::array<std::int64_t, 3> integers{std::numeric_limits<std::int64_t>::min(), 9'007'199'254'740'993,
                                               std::numeric_limits<std::int64_t>::max()};
    const NamedTensors source{{"weights", from_data({1.0F, -2.0F, 3.0F, 4.0F}, Shape{2, 2})},
                              {"indices", from_data(integers, Shape{3})}};
    std::stringstream stream;
    write_tensors(stream, source);
    const auto loaded = read_tensors(stream);

    ASSERT_EQ(loaded.size(), 2U);
    ASSERT_TRUE(loaded.contains("weights"));
    ASSERT_TRUE(loaded.contains("indices"));
    EXPECT_EQ(loaded.at("weights").shape(), (Shape{2, 2}));
    EXPECT_EQ(loaded.at("weights").dtype(), DType::Float32);
    EXPECT_EQ(loaded.at("weights").device(), Device::cpu());
    EXPECT_EQ(to_vector(loaded.at("weights")), (std::vector<float>{1.0F, -2.0F, 3.0F, 4.0F}));
    EXPECT_EQ(loaded.at("indices").shape(), (Shape{3}));
    EXPECT_EQ(loaded.at("indices").dtype(), DType::Int64);
    EXPECT_EQ(to_vector<std::int64_t>(loaded.at("indices")),
              (std::vector<std::int64_t>{integers.begin(), integers.end()}));
}

TEST(TensorPersistenceTest, SupportsEmptyArchivesScalarsAndEmptyTensors)
{
    std::stringstream empty_stream;
    write_tensors(empty_stream, {});
    EXPECT_TRUE(read_tensors(empty_stream).empty());

    std::stringstream stream;
    write_tensors(stream,
                  {{"scalar", full(Shape{}, -3.25F)}, {"empty", from_data(std::span<const float>{}, Shape{2, 0, 3})}});
    const auto loaded = read_tensors(stream);
    EXPECT_EQ(loaded.at("scalar").shape(), Shape{});
    EXPECT_FLOAT_EQ(item(loaded.at("scalar")), -3.25F);
    EXPECT_EQ(loaded.at("empty").shape(), (Shape{2, 0, 3}));
    EXPECT_TRUE(to_vector(loaded.at("empty")).empty());
}

TEST(TensorPersistenceTest, PreservesOpaqueNamesAndDistinctEntriesWithTheSameValue)
{
    const Tensor value = full(Shape{}, 7.0F);
    const std::string name{"a\0\xff", 3};
    std::stringstream stream;
    write_tensors(stream, {{"", value}, {name, value}});
    const auto loaded = read_tensors(stream);

    ASSERT_EQ(loaded.size(), 2U);
    EXPECT_FLOAT_EQ(item(loaded.at("")), 7.0F);
    EXPECT_FLOAT_EQ(item(loaded.at(name)), 7.0F);
    EXPECT_FLOAT_EQ(item(value), 7.0F);
}

TEST(TensorPersistenceTest, MaterializesLazyTensorsAndWritesViewsInLogicalOrder)
{
    const Tensor matrix = from_data({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, Shape{2, 3});
    std::stringstream stream;
    write_tensors(stream, {{"view", transpose(matrix)}, {"lazy", full(Shape{2}, 3.0F)}});
    const auto loaded = read_tensors(stream);

    EXPECT_EQ(loaded.at("view").shape(), (Shape{3, 2}));
    EXPECT_EQ(to_vector(loaded.at("view")), (std::vector<float>{1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F}));
    EXPECT_EQ(to_vector(loaded.at("lazy")), (std::vector<float>{3.0F, 3.0F}));
}

TEST(TensorPersistenceTest, PreservesFloatingPointBitPatterns)
{
    const std::array<std::uint32_t, 4> patterns{0x80000000, 0x7f800000, 0xff800000, 0x7fc01234};
    std::vector<float> values;
    for (const auto pattern : patterns)
    {
        values.push_back(std::bit_cast<float>(pattern));
    }
    std::stringstream stream;
    write_tensors(stream, {{"values", from_data(values, Shape{4})}});
    const auto loaded = to_vector(read_tensors(stream).at("values"));

    ASSERT_EQ(loaded.size(), patterns.size());
    for (std::size_t i = 0; i < patterns.size(); ++i)
    {
        EXPECT_EQ(std::bit_cast<std::uint32_t>(loaded[i]), patterns[i]);
    }
}

TEST(TensorPersistenceTest, HandlesPayloadsLargerThanOneIoChunk)
{
    const std::vector<float> values(300'000, 1.5F);
    std::stringstream stream;
    write_tensors(stream, {{"values", from_data(values, Shape{300'000})}});
    EXPECT_EQ(to_vector(read_tensors(stream).at("values")), values);
}

TEST(TensorPersistenceTest, ResolvesIndicesIndependentlyOfNameOrder)
{
    std::stringstream stream;
    detail::write_archive(stream, {{full(Shape{}, 2.0F), full(Shape{}, 3.0F)}, {{"first", 1}, {"second", 0}}});
    const auto loaded = read_tensors(stream);
    EXPECT_FLOAT_EQ(item(loaded.at("first")), 3.0F);
    EXPECT_FLOAT_EQ(item(loaded.at("second")), 2.0F);
}

TEST(TensorPersistenceTest, RejectsAliasesAndUnboundRecords)
{
    const Tensor value = full(Shape{}, 1.0F);
    for (const auto &data : std::vector<detail::ArchiveData>{
             {{value, value}, {{"first", 0}, {"second", 0}}}, {{value}, {}}, {{value}, {{"first", 0}, {"second", 0}}}})
    {
        std::stringstream stream;
        detail::write_archive(stream, data);
        EXPECT_THROW((void)read_tensors(stream), std::invalid_argument);
    }
}

TEST(TensorPersistenceTest, ReportsUnavailableTargetDevices)
{
    std::stringstream stream;
    write_tensors(stream, {{"value", full(Shape{}, 1.0F)}});
    EXPECT_THROW((void)read_tensors(stream, Device::cpu(1)), std::runtime_error);
}
} // namespace minitensor::persistence::test
