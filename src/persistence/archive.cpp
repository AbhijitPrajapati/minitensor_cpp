#include "minitensor/persistence/detail/archive.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
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
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::persistence::detail
{

namespace
{

// Bound each operation by streamsize while allowing arbitrarily large spans.
constexpr auto byte_chunk_size =
    static_cast<std::size_t>(std::min<std::uintmax_t>(1 << 20, std::numeric_limits<std::streamsize>::max()));

void write_bytes(std::ostream &output, std::span<const std::byte> bytes)
{
    if (!output)
    {
        throw std::runtime_error{"cannot write archive bytes: stream is not ready"};
    }

    while (!bytes.empty())
    {
        const auto count = std::min(bytes.size(), byte_chunk_size);
        output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(count));
        if (!output)
        {
            throw std::runtime_error{"failed to write archive bytes"};
        }
        bytes = bytes.subspan(count);
    }
}

void read_bytes(std::istream &input, std::span<std::byte> bytes)
{
    if (!input)
    {
        throw std::runtime_error{"cannot read archive bytes: stream is not ready"};
    }

    while (!bytes.empty())
    {
        const auto count = std::min(bytes.size(), byte_chunk_size);
        input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(count));
        if (!input)
        {
            throw std::runtime_error{"failed to read archive bytes: truncated input or stream error"};
        }
        bytes = bytes.subspan(count);
    }
}

// Read and write little-endian integers.

template <std::unsigned_integral T>
void write_integer(std::ostream &output, T value)
{
    static_assert(std::numeric_limits<T>::digits == sizeof(T) * 8,
                  "archive integers require eight-bit bytes and no padding bits");

    std::array<std::byte, sizeof(T)> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i)
    {
        // Shift to the right and mask for the final 8 bits to extract the least significant remaining byte.
        bytes[i] = static_cast<std::byte>((value >> (8 * i)) & 0xff);
    }
    write_bytes(output, bytes);
}

template <std::unsigned_integral T>
T read_integer(std::istream &input)
{
    static_assert(std::numeric_limits<T>::digits == sizeof(T) * 8,
                  "archive integers require eight-bit bytes and no padding bits");

    std::array<std::byte, sizeof(T)> bytes{};
    read_bytes(input, bytes);

    T value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i)
    {
        // Shift the byte to the left before adding to the value to reverse little-endian encoding.
        value |= static_cast<T>(std::to_integer<unsigned char>(bytes[i])) << (8 * i);
    }
    return value;
}

// Bound individual names before allocating or emitting their length prefix.
constexpr std::uint64_t max_string_bytes = 1 << 20;

void write_string(std::ostream &output, std::string_view value)
{
    if (value.size() > max_string_bytes)
    {
        throw std::length_error{"archive string exceeds the size limit"};
    }

    write_integer<std::uint64_t>(output, static_cast<std::uint64_t>(value.size()));
    write_bytes(output, std::as_bytes(std::span{value.data(), value.size()}));
}

std::string read_string(std::istream &input)
{
    const auto length = read_integer<std::uint64_t>(input);
    std::string value;
    if (length > max_string_bytes || !std::in_range<std::size_t>(length) || length > value.max_size())
    {
        throw std::length_error{"archive string exceeds the size limit"};
    }

    value.resize(static_cast<std::size_t>(length));
    read_bytes(input, std::as_writable_bytes(std::span{value.data(), value.size()}));
    return value;
}

// Wire tags are independent of the numeric values of the public DType enum.
enum class DTypeTag : std::uint8_t
{
    Float32 = 1,
    Int64 = 2,
};

void write_dtype(std::ostream &output, DType dtype)
{
    DTypeTag tag;
    switch (dtype)
    {
    case DType::Float32:
        tag = DTypeTag::Float32;
        break;
    case DType::Int64:
        tag = DTypeTag::Int64;
        break;
    default:
        throw std::invalid_argument{"unsupported archive dtype"};
    }
    write_integer<std::uint8_t>(output, static_cast<std::uint8_t>(tag));
}

DType read_dtype(std::istream &input)
{
    const auto tag = static_cast<DTypeTag>(read_integer<std::uint8_t>(input));
    switch (tag)
    {
    case DTypeTag::Float32:
        return DType::Float32;
    case DTypeTag::Int64:
        return DType::Int64;
    }
    throw std::invalid_argument{"unsupported archive dtype tag"};
}

// At most 16 KiB of encoded extents per tensor.
constexpr std::uint32_t max_tensor_rank = 2048;

void write_shape(std::ostream &output, const Shape &shape)
{
    if (shape.rank() > max_tensor_rank)
    {
        throw std::length_error{"archive tensor rank exceeds the size limit"};
    }

    write_integer(output, static_cast<std::uint32_t>(shape.rank()));
    for (Extent extent : shape)
    {
        write_integer(output, static_cast<std::uint64_t>(extent));
    }
}

Shape read_shape(std::istream &input)
{
    const auto rank = read_integer<std::uint32_t>(input);
    if (rank > max_tensor_rank)
    {
        throw std::length_error{"archive tensor rank exceeds the size limit"};
    }

    std::vector<Extent> dimensions;
    dimensions.reserve(rank);
    for (std::uint32_t i = 0; i < rank; ++i)
    {
        const auto extent = read_integer<std::uint64_t>(input);
        if (!std::in_range<Extent>(extent) || !std::in_range<std::size_t>(extent))
        {
            throw std::overflow_error{"archive tensor extent is not representable"};
        }
        dimensions.push_back(static_cast<Extent>(extent));
    }
    return Shape{std::move(dimensions)};
}

// Version 1 (all integers little-endian):
//   magic[8], version:u32, tensor_count:u64, binding_count:u64
//   tensor_count * (dtype:u8, rank:u32, extents:rank*u64)
//   binding_count * (name:string, tensor_index:u64)
//   zero padding to an eight-byte boundary, then payloads in tensor order.

std::uint64_t tensor_byte_count(const Shape &shape, DType dtype)
{
    const auto element_size = dtype_size(dtype);
    if (shape.numel() > std::numeric_limits<std::size_t>::max() / element_size)
    {
        throw std::overflow_error{"archive tensor byte count exceeds size_t"};
    }
    const auto size = shape.numel() * element_size;
    if (!std::in_range<std::uint64_t>(size))
    {
        throw std::overflow_error{"archive tensor byte count exceeds the integer range"};
    }
    return static_cast<std::uint64_t>(size);
}

// Reverse payload bytes for big-endian systems. Preserves bytes otherwise.
template <TensorElement T>
void convert_payload_byte_order(std::span<T> values)
{
    static_assert(std::endian::native == std::endian::little || std::endian::native == std::endian::big,
                  "archive payloads require a uniform byte order");
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "archive Float32 payloads require IEEE 754 binary32");
    if constexpr (std::endian::native == std::endian::big)
    {
        for (auto &value : values)
        {
            auto bytes = std::as_writable_bytes(std::span<T>{&value, 1});
            std::reverse(bytes.begin(), bytes.end());
        }
    }
}

template <TensorElement T>
void write_typed_payload(std::ostream &output, const Tensor &tensor)
{
    auto values = to_vector<T>(tensor);
    convert_payload_byte_order(std::span{values});
    write_bytes(output, std::as_bytes(std::span{values}));
}

void write_tensor_payload(std::ostream &output, const Tensor &tensor)
{
    switch (tensor.dtype())
    {
    case DType::Float32:
        return write_typed_payload<float>(output, tensor);
    case DType::Int64:
        return write_typed_payload<std::int64_t>(output, tensor);
    }
    throw std::invalid_argument{"unsupported archive dtype"};
}

template <TensorElement T>
Tensor read_typed_payload(std::istream &input, const Shape &shape, Device device)
{
    std::vector<T> values(shape.numel());
    read_bytes(input, std::as_writable_bytes(std::span{values}));
    convert_payload_byte_order(std::span{values});
    return from_data<T>(values, shape, device);
}

Tensor read_tensor_payload(std::istream &input, DType dtype, const Shape &shape, Device device)
{
    switch (dtype)
    {
    case DType::Float32:
        return read_typed_payload<float>(input, shape, device);
    case DType::Int64:
        return read_typed_payload<std::int64_t>(input, shape, device);
    }
    throw std::invalid_argument{"unsupported archive dtype"};
}

// Pad to eight-byte boundary.
std::size_t pad_size(std::uint64_t header_size)
{
    return static_cast<std::size_t>((8 - header_size % 8) % 8);
}

void write_padding(std::ostream &output, std::uint64_t header_size)
{
    const std::array<std::byte, 8> zeros{};
    write_bytes(output, std::span{zeros}.first(pad_size(header_size)));
}

void read_padding(std::istream &input, std::uint64_t header_size)
{
    std::array<std::byte, 8> padding{};
    const auto bytes = std::span{padding}.first(pad_size(header_size));
    read_bytes(input, bytes);
    // Validate 0 bytes for padding.
    if (std::any_of(bytes.begin(), bytes.end(), [](std::byte value) { return value != std::byte{0}; }))
    {
        throw std::invalid_argument{"archive padding must be zero"};
    }
}

void require_end_of_stream(std::istream &input)
{
    try
    {
        if (input.peek() != std::char_traits<char>::eof())
        {
            throw std::invalid_argument{"unexpected bytes after archive payload"};
        }
    }
    catch (const std::ios_base::failure &)
    {
        // A normal EOF may throw when the caller enables eofbit exceptions.
        if (!input.eof() || input.bad())
        {
            throw;
        }
    }
    if (input.bad() || !input.eof())
    {
        throw std::runtime_error{"failed to check end of archive stream"};
    }
}

// Header and payload limits apply to the whole archive, not just each individual allocation.
void add_limited_size(std::uint64_t &total, std::uint64_t amount, std::uint64_t limit)
{
    if (amount > limit - total)
    {
        throw std::length_error{"archive exceeds the size limit"};
    }
    total += amount;
}

constexpr std::array<char, 8> archive_magic{'M', 'T', 'A', 'R', 'C', 'H', '\r', '\n'};
constexpr std::uint32_t archive_version = 1;
constexpr std::uint64_t archive_prefix_bytes = 8 + 4 + 8 + 8;
constexpr std::uint64_t max_header_bytes = 64 * 1024 * 1024;
constexpr std::uint64_t max_payload_bytes = std::uint64_t{1} << 30;

} // namespace

void write_archive(std::ostream &output, const ArchiveData &data)
{
    if (!std::in_range<std::uint64_t>(data.tensors.size()) || !std::in_range<std::uint64_t>(data.bindings.size()))
    {
        throw std::overflow_error{"archive record count exceeds the integer range"};
    }

    // Validate the header and payload sizes before writing or evaluating tensors.
    std::uint64_t header_size = archive_prefix_bytes;
    std::uint64_t payload_size = 0;
    for (const auto &tensor : data.tensors)
    {
        if (tensor.shape().rank() > max_tensor_rank)
        {
            throw std::length_error{"archive tensor rank exceeds the size limit"};
        }
        add_limited_size(header_size, 5 + 8 * tensor.shape().rank(), max_header_bytes);
        add_limited_size(payload_size, tensor_byte_count(tensor.shape(), tensor.dtype()), max_payload_bytes);
    }
    for (const auto &[name, index] : data.bindings)
    {
        if (name.size() > max_string_bytes)
        {
            throw std::length_error{"archive string exceeds the size limit"};
        }
        if (std::cmp_greater_equal(index, data.tensors.size()))
        {
            throw std::invalid_argument{"archive binding references an unknown tensor"};
        }
        add_limited_size(header_size, 16 + name.size(), max_header_bytes);
    }

    write_bytes(output, std::as_bytes(std::span{archive_magic}));
    write_integer(output, archive_version);
    write_integer<std::uint64_t>(output, static_cast<std::uint64_t>(data.tensors.size()));
    write_integer<std::uint64_t>(output, static_cast<std::uint64_t>(data.bindings.size()));
    for (const auto &tensor : data.tensors)
    {
        write_dtype(output, tensor.dtype());
        write_shape(output, tensor.shape());
    }
    for (const auto &[name, index] : data.bindings)
    {
        write_string(output, name);
        write_integer(output, index);
    }
    write_padding(output, header_size);
    for (const auto &tensor : data.tensors)
    {
        write_tensor_payload(output, tensor);
    }
}

ArchiveData read_archive(std::istream &input, Device device)
{
    std::array<char, archive_magic.size()> magic{};
    read_bytes(input, std::as_writable_bytes(std::span{magic}));
    if (magic != archive_magic)
    {
        throw std::invalid_argument{"unrecognized archive magic"};
    }
    if (read_integer<std::uint32_t>(input) != archive_version)
    {
        throw std::invalid_argument{"unsupported archive version"};
    }
    const auto tensor_count = read_integer<std::uint64_t>(input);
    const auto binding_count = read_integer<std::uint64_t>(input);

    // Validate minimum header and payload sizes before reading.
    std::uint64_t header_size = archive_prefix_bytes;
    if (!std::in_range<std::size_t>(tensor_count) || !std::in_range<std::size_t>(binding_count) ||
        tensor_count > (max_header_bytes - header_size) / 5)
    {
        throw std::length_error{"archive record count exceeds the size limit"};
    }
    add_limited_size(header_size, 5 * tensor_count, max_header_bytes);
    if (binding_count > (max_header_bytes - header_size) / 16)
    {
        throw std::length_error{"archive binding count exceeds the size limit"};
    }
    add_limited_size(header_size, 16 * binding_count, max_header_bytes);

    // Collect tensor metadata.
    struct TensorDescription
    {
        DType dtype;
        Shape shape;
    };
    std::vector<TensorDescription> descriptions;
    std::uint64_t payload_size = 0;
    for (std::uint64_t i = 0; i < tensor_count; ++i)
    {
        const auto dtype = read_dtype(input);
        auto shape = read_shape(input);
        add_limited_size(header_size, 8 * shape.rank(), max_header_bytes);
        add_limited_size(payload_size, tensor_byte_count(shape, dtype), max_payload_bytes);
        descriptions.push_back({dtype, std::move(shape)});
    }

    // Extract tensor bindings.
    ArchiveData data;
    for (std::uint64_t i = 0; i < binding_count; ++i)
    {
        auto name = read_string(input);
        add_limited_size(header_size, name.size(), max_header_bytes);
        const auto index = read_integer<std::uint64_t>(input);
        if (index >= tensor_count)
        {
            throw std::invalid_argument{"archive binding references an unknown tensor"};
        }
        if (!data.bindings.emplace(std::move(name), index).second)
        {
            throw std::invalid_argument{"duplicate archive binding name"};
        }
    }
    read_padding(input, header_size);

    // Read tensor payloads.
    data.tensors.reserve(descriptions.size());
    for (const auto &description : descriptions)
    {
        data.tensors.push_back(read_tensor_payload(input, description.dtype, description.shape, device));
    }
    require_end_of_stream(input);
    return data;
}

} // namespace minitensor::persistence::detail
