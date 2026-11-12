#include <minitensor/persistence/record.hpp>

#include <cstdint>
#include <ios>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/manipulation.hpp>

namespace minitensor::persistence::test
{
namespace
{
std::string write_archive(Record &record)
{
    std::ostringstream output(std::ios::binary);
    record.write(output);
    return std::move(output).str();
}

Record read_archive(const std::string &bytes, const Record::ReadOptions &options = {})
{
    std::istringstream input(bytes, std::ios::binary);
    return Record::read(input, options);
}
} // namespace

TEST(BinaryTest, RoundTripsNestedRecordsListsAndScalars)
{
    const std::string_view name{"a\0b", 3};
    const std::string text{"x\0y", 3};
    const double precise = 1.0 + std::numeric_limits<double>::epsilon();

    Record child;
    child.add(name, text);
    child.add("", "");
    Record record;
    record.add("values", Value::List{true, std::numeric_limits<std::int64_t>::min(), 1.25F, precise, std::move(child),
                                     Value::List{false}, Value::List{}});
    record.add("empty", Record{});

    const Record restored = read_archive(write_archive(record));
    EXPECT_TRUE(restored.get("empty").get<Record>().empty());
    const auto &values = restored.get("values").get<Value::List>();
    ASSERT_EQ(values.size(), 7U);
    EXPECT_TRUE(values[0].get<bool>());
    EXPECT_EQ(values[1].get<std::int64_t>(), std::numeric_limits<std::int64_t>::min());
    EXPECT_FLOAT_EQ(values[2].get<float>(), 1.25F);
    EXPECT_EQ(values[3].get<double>(), precise);
    EXPECT_EQ(values[4].get<Record>().get(name).get<std::string>(), text);
    EXPECT_TRUE(values[4].get<Record>().get("").get<std::string>().empty());
    const auto &nested = values[5].get<Value::List>();
    ASSERT_EQ(nested.size(), 1U);
    EXPECT_FALSE(nested.front().get<bool>());
    EXPECT_TRUE(values[6].get<Value::List>().empty());
}

TEST(BinaryTest, RoundTripsTensorValuesInLogicalOrder)
{
    Record record;
    record.add("view", transpose(from_data({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, Shape{2, 3})));
    constexpr std::int64_t integer = -9'007'199'254'740'993;
    record.add("scalar", from_data({integer}, Shape{}));
    record.add("empty", full(Shape{2, 0, 3}, 0.0F));

    const Record restored = read_archive(write_archive(record));
    const auto &view = restored.get("view").get<Tensor>();
    EXPECT_EQ(view.shape(), (Shape{3, 2}));
    EXPECT_EQ(view.dtype(), DType::Float32);
    EXPECT_EQ(view.device(), Device::cpu());
    EXPECT_EQ(to_vector(view), (std::vector<float>{1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F}));

    const auto &scalar = restored.get("scalar").get<Tensor>();
    EXPECT_EQ(scalar.shape(), Shape{});
    EXPECT_EQ(scalar.dtype(), DType::Int64);
    EXPECT_EQ(item<std::int64_t>(scalar), integer);

    const auto &empty = restored.get("empty").get<Tensor>();
    EXPECT_EQ(empty.shape(), (Shape{2, 0, 3}));
    EXPECT_TRUE(to_vector(empty).empty());
}

TEST(BinaryTest, RejectsInvalidHeadersAndTruncatedData)
{
    Record record;
    record.add("step", std::uint64_t{42});
    const auto bytes = write_archive(record);

    auto bad_magic = bytes;
    bad_magic.front() = 'X';
    EXPECT_THROW((void)read_archive(bad_magic), std::invalid_argument);

    auto bad_version = bytes;
    bad_version[6] = '\x02';
    EXPECT_THROW((void)read_archive(bad_version), std::invalid_argument);

    EXPECT_THROW((void)read_archive(bytes.substr(0, 6)), std::ios_base::failure);
    EXPECT_THROW((void)read_archive(bytes.substr(0, bytes.size() - 1)), std::ios_base::failure);
}

TEST(BinaryTest, RejectsUnknownValueTagsAndInvalidBooleans)
{
    Record record;
    record.add("flag", true);
    const auto bytes = write_archive(record);

    auto bad_kind = bytes;
    bad_kind[bad_kind.size() - 2] = '\x7F';
    EXPECT_THROW((void)read_archive(bad_kind), std::invalid_argument);

    auto bad_bool = bytes;
    bad_bool.back() = '\x02';
    EXPECT_THROW((void)read_archive(bad_bool), std::invalid_argument);
}
} // namespace minitensor::persistence::test
