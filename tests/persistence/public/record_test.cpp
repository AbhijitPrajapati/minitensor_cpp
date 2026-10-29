#include <minitensor/persistence/record.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <minitensor/autograd.hpp>
#include <minitensor/data.hpp>
#include <minitensor/evaluation.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/elementwise.hpp>
#include <minitensor/ops/reduction.hpp>

namespace minitensor::persistence::test
{
static_assert(std::is_copy_constructible_v<Record>);
static_assert(std::is_copy_assignable_v<Record>);
static_assert(std::is_nothrow_move_constructible_v<Value>);
static_assert(std::is_nothrow_move_assignable_v<Value>);
static_assert(std::same_as<decltype(std::declval<Record &>().begin()), Record::const_iterator>);
static_assert(std::is_convertible_v<bool, Value>);
static_assert(std::is_convertible_v<int, Value>);
static_assert(std::is_convertible_v<unsigned int, Value>);
static_assert(std::is_convertible_v<float, Value>);
static_assert(std::is_convertible_v<double, Value>);
static_assert(std::is_convertible_v<std::string, Value>);
static_assert(std::is_convertible_v<std::string_view, Value>);
static_assert(std::is_convertible_v<const char *, Value>);
static_assert(std::is_convertible_v<Tensor, Value>);
static_assert(std::is_convertible_v<Record, Value>);
static_assert(std::is_convertible_v<Value::List, Value>);
static_assert(!std::is_convertible_v<void *, Value>);
static_assert(!std::is_convertible_v<int *, Value>);
static_assert(!std::is_convertible_v<std::nullptr_t, Value>);
static_assert(!std::is_convertible_v<long double, Value>);

TEST(RecordTest, StartsEmptyAndReportsMissingFields)
{
    const Record record;
    EXPECT_TRUE(record.empty());
    EXPECT_EQ(record.size(), 0U);
    EXPECT_EQ(record.begin(), record.end());
    EXPECT_FALSE(record.contains("missing"));
    EXPECT_THROW((void)record.get("missing"), std::out_of_range);
    EXPECT_THROW((void)record.get("missing").get<std::uint64_t>(), std::out_of_range);
    EXPECT_THROW((void)record.get("missing").get<std::string>(), std::out_of_range);
    EXPECT_THROW((void)record.get("missing").get<Tensor>(), std::out_of_range);
    EXPECT_THROW((void)record.get("missing").get<Record>(), std::out_of_range);
    EXPECT_THROW((void)record.get("missing").get<Value::List>(), std::out_of_range);
}

TEST(RecordTest, PreservesScalarTypesAndPrecision)
{
    Record record;
    record.add("enabled", true);
    record.add("signed", std::numeric_limits<std::int64_t>::min());
    record.add("unsigned", std::numeric_limits<std::uint64_t>::max());
    record.add("float", 1.25F);
    const double precise = 1.0 + std::numeric_limits<double>::epsilon();
    record.add("double", precise);

    EXPECT_TRUE(record.get("enabled").get<bool>());
    EXPECT_EQ(record.get("signed").get<std::int64_t>(), std::numeric_limits<std::int64_t>::min());
    EXPECT_EQ(record.get("unsigned").get<std::uint64_t>(), std::numeric_limits<std::uint64_t>::max());
    EXPECT_FLOAT_EQ(record.get("float").get<float>(), 1.25F);
    EXPECT_EQ(record.get("double").get<double>(), precise);
    EXPECT_NE(record.get("double").get<double>(), static_cast<double>(static_cast<float>(precise)));
    EXPECT_EQ(record.get("enabled").kind(), Value::Kind::Bool);
    EXPECT_EQ(record.get("signed").kind(), Value::Kind::Int64);
    EXPECT_EQ(record.get("unsigned").kind(), Value::Kind::UInt64);
    EXPECT_EQ(record.get("float").kind(), Value::Kind::Float32);
    EXPECT_EQ(record.get("double").kind(), Value::Kind::Float64);
}

TEST(RecordTest, CanonicalizesIntegersBySignednessAndPreservesSpecialFloats)
{
    Record record;
    record.add("signed", -7);
    record.add("unsigned", 9U);
    record.add("infinity", std::numeric_limits<double>::infinity());
    record.add("nan", std::numeric_limits<float>::quiet_NaN());
    record.add("negative_zero", -0.0);

    EXPECT_EQ(record.get("signed").get<std::int64_t>(), -7);
    EXPECT_EQ(record.get("unsigned").get<std::uint64_t>(), 9U);
    EXPECT_TRUE(std::isinf(record.get("infinity").get<double>()));
    EXPECT_TRUE(std::isnan(record.get("nan").get<float>()));
    EXPECT_TRUE(std::signbit(record.get("negative_zero").get<double>()));
}

TEST(RecordTest, RejectsTypeMismatchesWithoutConverting)
{
    Record record;
    record.add("step", std::uint64_t{42});
    record.add("label", "training");

    EXPECT_TRUE(record.get("step").is<std::uint64_t>());
    EXPECT_FALSE(record.get("step").is<std::int64_t>());
    EXPECT_THROW((void)record.get("step").get<std::int64_t>(), std::logic_error);
    EXPECT_THROW((void)record.get("step").get<double>(), std::logic_error);
    EXPECT_THROW((void)record.get("step").get<bool>(), std::logic_error);
    EXPECT_THROW((void)record.get("step").get<std::string>(), std::logic_error);
    EXPECT_THROW((void)record.get("step").get<Tensor>(), std::logic_error);
    EXPECT_THROW((void)record.get("step").get<Record>(), std::logic_error);
    EXPECT_THROW((void)record.get("step").get<Value::List>(), std::logic_error);
    EXPECT_THROW((void)record.get("label").get<std::uint64_t>(), std::logic_error);
    EXPECT_EQ(record.size(), 2U);
}

TEST(RecordTest, OwnsNamesAndStringsAndTreatsNamesLiterally)
{
    Record record;
    std::string name = "model.weight";
    std::string text = "original";
    record.add(name, text);
    name.assign("changed");
    text.assign("changed");
    record.add("", "");
    record.add("binary", std::string_view{"a\0b", 3});

    EXPECT_TRUE(record.contains("model.weight"));
    EXPECT_FALSE(record.contains("model"));
    EXPECT_FALSE(record.contains("changed"));
    EXPECT_EQ(record.get("model.weight").get<std::string>(), "original");
    EXPECT_EQ(record.get("").get<std::string>(), "");
    EXPECT_EQ(record.get("binary").get<std::string>(), (std::string{"a\0b", 3}));
    EXPECT_EQ(record.get("binary").kind(), Value::Kind::String);
}

TEST(RecordTest, ImplicitTextConversionsOwnTheirContentsAndRejectNullPointers)
{
    Record record;
    char buffer[] = "original";
    const char *pointer = buffer;
    record.add("pointer", pointer);
    record.add("view", std::string_view{buffer});
    record.add("temporary", std::string{"temporary"});
    record.add("literal", "training");
    buffer[0] = 'X';

    EXPECT_EQ(record.get("pointer").get<std::string>(), "original");
    EXPECT_EQ(record.get("view").get<std::string>(), "original");
    EXPECT_EQ(record.get("temporary").get<std::string>(), "temporary");
    EXPECT_EQ(record.get("literal").get<std::string>(), "training");
    EXPECT_FALSE(record.get("literal").is<bool>());
    const char *null_text = nullptr;
    EXPECT_THROW(record.add("null", null_text), std::invalid_argument);
    EXPECT_FALSE(record.contains("null"));
    EXPECT_EQ(record.size(), 4U);
}

TEST(RecordTest, RejectsDuplicatesAcrossValueKindsAndRetainsOriginal)
{
    Record record;
    record.add("value", 3);
    EXPECT_THROW(record.add("value", 4), std::invalid_argument);
    EXPECT_THROW(record.add("value", "replacement"), std::invalid_argument);
    EXPECT_THROW(record.add("value", Record{}), std::invalid_argument);
    EXPECT_THROW(record.add("value", Value::List{}), std::invalid_argument);
    EXPECT_EQ(record.size(), 1U);
    EXPECT_EQ(record.get("value").get<std::int64_t>(), 3);
}

TEST(RecordTest, EnumeratesFieldsInNameOrder)
{
    Record record;
    record.add("z", 1);
    record.add("a.b", 2);
    record.add("a", 3);
    std::vector<std::string> names;
    for (const auto &[name, value] : record)
    {
        names.push_back(name);
        EXPECT_TRUE(value.is<std::int64_t>());
    }
    EXPECT_EQ(names, (std::vector<std::string>{"a", "a.b", "z"}));
    EXPECT_FALSE(record.empty());
}

TEST(RecordTest, SupportsNestedRecordsAndHeterogeneousLists)
{
    Record child;
    child.add("epoch", std::uint64_t{12});
    Value::List nested{false, "batch"};
    Value::List values{42, 2.5, child, std::move(nested)};
    Record root;
    root.add("progress", child);
    root.add("values", std::move(values));
    root.add("empty", Value::List{});
    child.add("later", true);

    EXPECT_EQ(root.get("progress").get<Record>().get("epoch").get<std::uint64_t>(), 12U);
    EXPECT_FALSE(root.get("progress").get<Record>().contains("later"));
    EXPECT_TRUE(root.get("progress").is<Record>());
    EXPECT_EQ(root.get("progress").kind(), Value::Kind::Record);
    EXPECT_EQ(root.get("values").kind(), Value::Kind::List);
    const auto &list = root.get("values").get<Value::List>();
    ASSERT_EQ(list.size(), 4U);
    EXPECT_EQ(list[0].get<std::int64_t>(), 42);
    EXPECT_DOUBLE_EQ(list[1].get<double>(), 2.5);
    EXPECT_EQ(list[2].get<Record>().get("epoch").get<std::uint64_t>(), 12U);
    const auto &inner = list[3].get<Value::List>();
    ASSERT_EQ(inner.size(), 2U);
    EXPECT_FALSE(inner[0].get<bool>());
    EXPECT_EQ(inner[1].get<std::string>(), "batch");
    EXPECT_TRUE(root.get("empty").get<Value::List>().empty());
}

TEST(RecordTest, CopiesNestedStructureIndependentlyAndSupportsAssignmentAndMove)
{
    Record child;
    child.add("step", std::uint64_t{8});
    Record source;
    source.add("child", child);
    source.add("children", Value::List{child});
    Record copied = source;
    copied.add("extra", true);
    EXPECT_FALSE(source.contains("extra"));
    EXPECT_NE(&copied.get("child").get<Record>(), &source.get("child").get<Record>());
    EXPECT_NE(&copied.get("children").get<Value::List>()[0].get<Record>(),
              &source.get("children").get<Value::List>()[0].get<Record>());

    Record assigned;
    assigned.add("obsolete", "discarded");
    assigned = source;
    EXPECT_FALSE(assigned.contains("obsolete"));
    EXPECT_NE(&assigned.get("child").get<Record>(), &source.get("child").get<Record>());
    EXPECT_EQ(assigned.get("child").get<Record>().get("step").get<std::uint64_t>(), 8U);
    Record moved = std::move(copied);
    assigned = std::move(moved);
    EXPECT_TRUE(assigned.get("extra").get<bool>());
    EXPECT_EQ(assigned.get("children").get<Value::List>()[0].get<Record>().get("step").get<std::uint64_t>(), 8U);
    EXPECT_EQ(source.size(), 2U);
}

TEST(RecordTest, GenericValuesSupportCopyAssignmentAcrossKinds)
{
    Record child;
    child.add("name", "child");
    const Value original = child;
    Value value = 1;
    value = original;
    EXPECT_EQ(value.get<Record>().get("name").get<std::string>(), "child");
    EXPECT_NE(&value.get<Record>(), &original.get<Record>());
    value = Value::List{original, std::uint64_t{9}};
    Record root;
    root.add("custom", value);
    value = false;
    ASSERT_EQ(root.get("custom").get<Value::List>().size(), 2U);
    EXPECT_EQ(root.get("custom").get<Value::List>()[0].get<Record>().get("name").get<std::string>(), "child");
    EXPECT_EQ(root.get("custom").get<Value::List>()[1].get<std::uint64_t>(), 9U);
}

TEST(RecordTest, CapturesAndCopiesTensorHandlesWithoutEvaluation)
{
    Record source;
    {
        const Tensor unavailable = full(Shape{2}, 1.0F, TensorOptions{DType::Float32, Device::cpu(1)});
        source.add("lazy", unavailable);
        source.add("tensors", Value::List{unavailable});
    }
    const Record copied = source;
    EXPECT_EQ(copied.get("lazy").get<Tensor>().shape(), (Shape{2}));
    EXPECT_EQ(copied.get("lazy").get<Tensor>().dtype(), DType::Float32);
    EXPECT_EQ(copied.get("lazy").get<Tensor>().device(), Device::cpu(1));
    EXPECT_EQ(copied.get("lazy").kind(), Value::Kind::Tensor);
    EXPECT_EQ(copied.get("tensors").get<Value::List>()[0].get<Tensor>().device(), Device::cpu(1));
    // An accidental evaluation during insertion or copying would already have thrown.
    EXPECT_THROW(eval(copied.get("lazy").get<Tensor>()), std::runtime_error);
}

TEST(RecordTest, RetainsTensorValuesAndAutogradConnections)
{
    const Tensor input = full(Shape{2}, 3.0F);
    Record record;
    record.add("result", input * input);
    const Record copied = record;

    EXPECT_EQ(to_vector(copied.get("result").get<Tensor>()), (std::vector<float>{9.0F, 9.0F}));
    EXPECT_EQ(to_vector(grad(sum(copied.get("result").get<Tensor>()), input).front()),
              (std::vector<float>{6.0F, 6.0F}));
    // Both records retain the same graph value, rather than introducing new leaves.
    EXPECT_EQ(to_vector(grad(sum(copied.get("result").get<Tensor>()), record.get("result").get<Tensor>()).front()),
              (std::vector<float>{1.0F, 1.0F}));
}
} // namespace minitensor::persistence::test
