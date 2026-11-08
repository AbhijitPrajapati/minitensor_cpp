#include "binary.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <istream>
#include <limits>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <minitensor/data.hpp>
#include <minitensor/persistence/record.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::persistence::detail
{

namespace
{

constexpr std::string_view magic = "MTARCH";

constexpr std::size_t max_nesting_depth = 64;

std::size_t child_depth(std::size_t remaining_depth)
{
    if (remaining_depth == 0)
    {
        throw std::length_error{"archive nesting depth exceeds limit"};
    }
    return remaining_depth - 1;
}

enum class WireKind : std::uint8_t
{
    Bool = 0x01,
    Int64 = 0x02,
    UInt64 = 0x03,
    Float32 = 0x04,
    Float64 = 0x05,
    String = 0x06,
    Tensor = 0x07,
    Record = 0x08,
    List = 0x09,
};

enum class WireDType : std::uint8_t
{
    Float32 = 0x01,
    Int64 = 0x02,
};

// Safely write bytes to output stream.
void write_bytes(std::ostream &output, std::span<const std::byte> bytes)
{
    if (bytes.empty())
    {
        return;
    }
    if (!std::in_range<std::streamsize>(bytes.size()))
    {
        throw std::length_error{"archive write is too large"};
    }

    output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output)
    {
        throw std::ios_base::failure{"failed to write archive bytes"};
    }
}

// Safely read bytes from input stream.
void read_bytes(std::istream &input, std::span<std::byte> bytes)
{
    if (bytes.empty())
    {
        return;
    }
    if (!std::in_range<std::streamsize>(bytes.size()))
    {
        throw std::length_error{"archive read is too large"};
    }

    input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input)
    {
        throw std::ios_base::failure{"failed to read archive bytes"};
    }
}

void write_u8(std::ostream &output, std::uint8_t value)
{
    const std::array bytes{static_cast<std::byte>(value)};
    write_bytes(output, bytes);
}

std::uint8_t read_u8(std::istream &input)
{
    std::array<std::byte, 1> bytes{};
    read_bytes(input, bytes);
    return std::to_integer<std::uint8_t>(bytes[0]);
}

void write_u32(std::ostream &output, std::uint32_t value)
{
    std::array<std::byte, 4> bytes{};
    for (auto &byte : bytes)
    {
        // Keep rightmost 8 bits.
        byte = static_cast<std::byte>(value & 0xFFu);
        value >>= 8; // Shift to get the next 8 bits next time.
    }
    write_bytes(output, bytes);
}

std::uint32_t read_u32(std::istream &input)
{
    std::array<std::byte, 4> bytes{};
    read_bytes(input, bytes);

    std::uint32_t value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i)
    {
        // Add bytes right-to-left.
        const auto byte = std::to_integer<std::uint32_t>(bytes[i]);
        value |= byte << (8 * i);
    }
    return value;
}

void write_u64(std::ostream &output, std::uint64_t value)
{
    std::array<std::byte, 8> bytes{};
    for (auto &byte : bytes)
    {
        // Keep rightmost 8 bits.
        byte = static_cast<std::byte>(value & 0xFFu);
        value >>= 8; // Shift to get the next 8 bit next time.
    }
    write_bytes(output, bytes);
}

std::uint64_t read_u64(std::istream &input)
{
    std::array<std::byte, 8> bytes{};
    read_bytes(input, bytes);

    std::uint64_t value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i)
    {
        // Add bytes right-to-left.
        const auto byte = std::to_integer<std::uint64_t>(bytes[i]);
        value |= byte << (8 * i);
    }
    return value;
}

// Safely convert size_t to uint32 and write.
void write_size(std::ostream &output, std::size_t size)
{
    if (size > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::overflow_error("size is too large to fit in uint32");
    }
    write_u32(output, static_cast<std::uint32_t>(size));
}

// Safely read a uint32 size and convert to size_t.
std::size_t read_size(std::istream &input, std::size_t limit)
{
    const std::uint32_t size = read_u32(input);
    if (!std::in_range<std::size_t>(size) || size > limit)
    {
        throw std::length_error{"size exceeds read limit"};
    }
    return static_cast<std::size_t>(size);
}

// Write string size and bytes.
void write_string(std::ostream &output, std::string_view string)
{
    write_size(output, string.size());
    write_bytes(output, std::as_bytes(std::span{string}));
}

// Read string size and bytes.
std::string read_string(std::istream &input, std::size_t max_bytes)
{
    const std::size_t size = read_size(input, max_bytes);
    std::string result(size, '\0');
    read_bytes(input, std::as_writable_bytes(std::span{result.data(), result.size()}));
    return result;
}

// Write tensor DType, rank, and dimensions.
void write_tensor_info(std::ostream &output, WireDType dtype, const Tensor &tensor)
{
    write_u8(output, static_cast<std::uint8_t>(dtype));
    write_size(output, tensor.rank());
    for (Extent extent : tensor.shape())
    {
        write_u64(output, static_cast<std::uint64_t>(extent));
    }
}

// Write tensor info and elements.
void write_tensor(std::ostream &output, const Tensor &tensor)
{
    switch (tensor.dtype())
    {
    case minitensor::DType::Float32:
        write_tensor_info(output, WireDType::Float32, tensor);
        for (float element : to_vector<float>(tensor))
        {
            write_u32(output, std::bit_cast<std::uint32_t>(element));
        }
        return;
    case DType::Int64:
        write_tensor_info(output, WireDType::Int64, tensor);
        for (std::int64_t element : to_vector<std::int64_t>(tensor))
        {
            write_u64(output, static_cast<std::uint64_t>(element));
        }
        return;
    }
}

Shape read_tensor_shape(std::istream &input, std::size_t max_rank)
{
    const std::size_t rank = read_size(input, max_rank);
    std::vector<Extent> dimensions;
    dimensions.reserve(rank);

    for (std::size_t axis = 0; axis < rank; ++axis)
    {
        const std::uint64_t extent = read_u64(input);
        if (!std::in_range<Extent>(extent) || !std::in_range<std::size_t>(extent))
        {
            throw std::length_error{"archive tensor extent is too large"};
        }
        dimensions.push_back(static_cast<Extent>(extent));
    }
    return Shape(std::move(dimensions));
}

Tensor read_tensor(std::istream &input, Device device, std::size_t max_rank, std::size_t max_tensor_bytes)
{
    const auto dtype = static_cast<WireDType>(read_u8(input));

    Shape shape = read_tensor_shape(input, max_rank);
    const std::size_t numel = shape.numel();

    switch (dtype)
    {
    case WireDType::Float32: {
        if (numel > max_tensor_bytes / 4)
        {
            throw std::length_error{"archive tensor exceeds byte limit"};
        }
        std::vector<float> elements(numel);
        for (float &element : elements)
        {
            element = std::bit_cast<float>(read_u32(input));
        }
        return from_data(elements, std::move(shape), device);
    }

    case WireDType::Int64: {
        if (numel > max_tensor_bytes / 8)
        {
            throw std::length_error{"archive tensor exceeds byte limit"};
        }
        std::vector<std::int64_t> elements(numel);
        for (std::int64_t &element : elements)
        {
            element = std::bit_cast<std::int64_t>(read_u64(input));
        }
        return from_data(elements, std::move(shape), device);
    }
    }
    throw std::invalid_argument{"unknown archive tensor dtype"};
}

void write_subrecord(std::ostream &output, const Record &record, std::size_t depth);
Record read_subrecord(std::istream &input, const Record::ReadOptions &options, std::size_t depth);

void write_value(std::ostream &output, const Value &value, std::size_t depth)
{
    switch (value.kind())
    {
    case Value::Kind::Bool:
        write_u8(output, static_cast<std::uint8_t>(WireKind::Bool));
        write_u8(output, value.get<bool>() ? 1 : 0);
        return;
    case Value::Kind::String:
        write_u8(output, static_cast<std::uint8_t>(WireKind::String));
        write_string(output, value.get<std::string>());
        return;
    case Value::Kind::UInt64:
        write_u8(output, static_cast<std::uint8_t>(WireKind::UInt64));
        write_u64(output, value.get<std::uint64_t>());
        return;
    case Value::Kind::Int64:
        write_u8(output, static_cast<std::uint8_t>(WireKind::Int64));
        write_u64(output, static_cast<std::uint64_t>(value.get<std::int64_t>()));
        return;
    case Value::Kind::Float32:
        write_u8(output, static_cast<std::uint8_t>(WireKind::Float32));
        write_u32(output, std::bit_cast<std::uint32_t>(value.get<float>()));
        return;
    case Value::Kind::Float64:
        write_u8(output, static_cast<std::uint8_t>(WireKind::Float64));
        write_u64(output, std::bit_cast<std::uint64_t>(value.get<double>()));
        return;
    case Value::Kind::Tensor:
        write_u8(output, static_cast<std::uint8_t>(WireKind::Tensor));
        write_tensor(output, value.get<Tensor>());
        return;
    case Value::Kind::Record: {
        const auto remaining_depth = child_depth(depth);
        write_u8(output, static_cast<std::uint8_t>(WireKind::Record));
        write_subrecord(output, value.get<Record>(), remaining_depth);
        return;
    }
    case Value::Kind::List:
        const auto remaining_depth = child_depth(depth);
        write_u8(output, static_cast<std::uint8_t>(WireKind::List));
        const auto &list = value.get<Value::List>();
        write_size(output, list.size());
        for (const Value &v : list)
        {
            write_value(output, v, remaining_depth);
        }
        return;
    }
}

Value read_value(std::istream &input, const Record::ReadOptions &options, std::size_t depth)
{
    const auto kind = static_cast<WireKind>(read_u8(input));
    switch (kind)
    {
    case WireKind::Bool: {
        const std::uint8_t bool_byte = read_u8(input);
        if (bool_byte > 1)
        {
            throw std::invalid_argument{"invalid boolean"};
        }
        return Value(bool_byte != 0);
    }
    case WireKind::String:
        return Value(read_string(input, options.max_string_bytes));
    case WireKind::UInt64:
        return Value(read_u64(input));
    case WireKind::Int64:
        return Value(static_cast<std::int64_t>(read_u64(input)));
    case WireKind::Float32:
        return Value(std::bit_cast<float>(read_u32(input)));
    case WireKind::Float64:
        return Value(std::bit_cast<double>(read_u64(input)));
    case WireKind::Tensor:
        return Value(read_tensor(input, options.device, options.max_tensor_rank, options.max_tensor_payload_bytes));
    case WireKind::Record:
        return Value(read_subrecord(input, options, child_depth(depth)));
    case WireKind::List:
        const auto remaining_depth = child_depth(depth);
        const std::size_t size = read_size(input, options.max_values_per_list);
        std::vector<Value> values;
        values.reserve(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            values.push_back(Value(read_value(input, options, remaining_depth)));
        }
        return Value(std::move(values));
    }

    throw std::invalid_argument{"unknown archive value kind"};
}

void write_subrecord(std::ostream &output, const Record &record, std::size_t depth)
{
    write_size(output, record.size());
    for (const auto &[field, value] : record)
    {
        write_string(output, field);
        write_value(output, value, depth);
    }
}

Record read_subrecord(std::istream &input, const Record::ReadOptions &options, std::size_t depth)
{
    const std::size_t field_count = read_size(input, options.max_fields_per_record);
    Record record;
    for (std::size_t i = 0; i < field_count; ++i)
    {
        const std::string name = read_string(input, options.max_field_bytes);
        record.add(name, read_value(input, options, depth));
    }
    return record;
}

} // namespace

void write_record(std::ostream &output, const Record &record)
{
    write_bytes(output, std::as_bytes(std::span{magic}));
    write_u8(output, 1);
    write_subrecord(output, record, max_nesting_depth);
}

Record read_record(std::istream &input, const Record::ReadOptions &options)
{
    std::string archive_magic(magic.size(), '\0');
    read_bytes(input, std::as_writable_bytes(std::span{archive_magic.data(), archive_magic.size()}));
    if (archive_magic != magic)
    {
        throw std::invalid_argument{"not a MiniTensor archive"};
    }

    const std::uint8_t archive_version = read_u8(input);
    if (archive_version != 1)
    {
        throw std::invalid_argument{"unsupported archive version"};
    }
    return read_subrecord(input, options, max_nesting_depth);
}
} // namespace minitensor::persistence::detail