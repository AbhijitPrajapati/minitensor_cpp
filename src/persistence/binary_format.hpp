#pragma once

#include <array>
#include <bit>
#include <climits>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

#include <minitensor/persistence/record.hpp>

namespace minitensor::persistence::detail::binary
{
static_assert(CHAR_BIT == 8);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
static_assert(std::endian::native == std::endian::little || std::endian::native == std::endian::big);

inline constexpr std::array<std::byte, 8> magic{std::byte{'M'}, std::byte{'T'}, std::byte{'A'},  std::byte{'R'},
                                                std::byte{'C'}, std::byte{'H'}, std::byte{'\r'}, std::byte{'\n'}};
inline constexpr std::array<std::byte, 4> end_marker{std::byte{'M'}, std::byte{'E'}, std::byte{'N'}, std::byte{'D'}};
inline constexpr std::uint32_t version = 2;
inline constexpr std::size_t header_bytes = 28;
inline constexpr std::size_t frame_bytes = 64 * 1024;
inline constexpr std::size_t spool_memory_bytes = 1024 * 1024;
inline constexpr std::size_t max_depth = 64;
inline constexpr std::uint64_t max_archive_bytes = std::uint64_t{4} * 1024 * 1024 * 1024;
inline constexpr std::uint64_t max_allocation_bytes = std::uint64_t{1} * 1024 * 1024 * 1024;
inline constexpr std::uint64_t max_values = 1'000'000;

// These internal limits govern both reading and writing.
inline constexpr std::size_t max_fields = 1'000'000;
inline constexpr std::size_t max_list_values = 1'000'000;
inline constexpr std::size_t max_text_bytes = 1024 * 1024;
inline constexpr std::size_t max_tensor_rank = 64;
inline constexpr std::size_t max_tensor_bytes = 256 * 1024 * 1024;

enum class Kind : std::uint8_t
{
    Bool = 0x01,
    Int64 = 0x02,
    UInt64 = 0x03,
    Float32 = 0x04,
    Float64 = 0x05,
    String = 0x06,
    Tensor = 0x07,
    Record = 0x08,
    List = 0x09
};
enum class TensorType : std::uint8_t
{
    Float32 = 1,
    Int64 = 2
};

template <std::unsigned_integral T>
constexpr void store_le(std::span<std::byte> destination, T value)
{
    for (std::size_t index = 0; index < sizeof(T); ++index)
    {
        destination[index] = static_cast<std::byte>(value & 0xffU);
        value >>= 8;
    }
}

template <std::unsigned_integral T>
constexpr T load_le(std::span<const std::byte> source)
{
    static_assert(sizeof(T) <= sizeof(std::uint64_t));
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < sizeof(T); ++index)
    {
        value |= static_cast<std::uint64_t>(std::to_integer<unsigned int>(source[index])) << (8 * index);
    }
    return static_cast<T>(value);
}

// Reflected CRC-32C, initial/final XOR 0xffffffff, as used by RFC 3720.
inline constexpr auto crc_table = [] {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t index = 0; index < table.size(); ++index)
    {
        auto value = index;
        for (unsigned int bit = 0; bit < 8; ++bit)
        {
            value = (value >> 1) ^ ((value & 1U) ? 0x82f63b78U : 0U);
        }
        table[index] = value;
    }
    return table;
}();

class Checksum
{
  public:
    constexpr void update(std::span<const std::byte> bytes)
    {
        for (std::byte byte : bytes)
        {
            value_ = crc_table[(value_ ^ std::to_integer<std::uint32_t>(byte)) & 0xffU] ^ (value_ >> 8);
        }
    }

    [[nodiscard]] constexpr std::uint32_t value() const noexcept
    {
        return value_ ^ 0xffffffffU;
    }

  private:
    std::uint32_t value_{0xffffffffU};
};

inline constexpr auto crc32c(std::span<const std::byte> bytes)
{
    Checksum checksum;
    checksum.update(bytes);
    return checksum.value();
}

static_assert(crc32c(std::array{std::byte{'1'}, std::byte{'2'}, std::byte{'3'}, std::byte{'4'}, std::byte{'5'},
                                std::byte{'6'}, std::byte{'7'}, std::byte{'8'}, std::byte{'9'}}) == 0xe3069283U);

inline void validate_archive_size(std::uint64_t payload_bytes)
{
    const auto frames = payload_bytes / frame_bytes + (payload_bytes % frame_bytes != 0);
    const auto overhead = header_bytes + 8 + frames * 4;
    if (payload_bytes > max_archive_bytes || overhead > max_archive_bytes - payload_bytes)
    {
        throw std::length_error{"archive exceeds the encoded byte budget"};
    }
}

// Conservative allocation accounting, not a measurement of allocator/driver RSS.
// Tensor charges include staging plus destination, and container charges include
// capacity growth. Charges are cumulative, so freed staging never grants more budget.
class Budget
{
  public:
    void record(std::size_t count)
    {
        nodes(count);
        bytes(sizeof(Record) +
              static_cast<std::uint64_t>(count) * (sizeof(Record::Fields::value_type) + 8 * sizeof(void *)));
    }

    void list(std::size_t count)
    {
        nodes(count);
        bytes(sizeof(Value::List) + 2 * static_cast<std::uint64_t>(count) * sizeof(Value));
    }

    void text(std::size_t count)
    {
        bytes(static_cast<std::uint64_t>(count) + 1 + 32);
    }

    void tensor(std::size_t count, std::size_t rank)
    {
        bytes(2 * static_cast<std::uint64_t>(count) + 512 + 2 * rank * sizeof(std::int64_t));
    }

  private:
    void bytes(std::uint64_t count)
    {
        if (count > max_allocation_bytes - allocated_)
        {
            throw std::length_error{"archive exceeds the decoded allocation budget"};
        }
        allocated_ += count;
    }

    void nodes(std::uint64_t count)
    {
        if (count > max_values - values_)
        {
            throw std::length_error{"archive exceeds the total value budget"};
        }
        values_ += count;
    }

    std::uint64_t allocated_{0};
    std::uint64_t values_{0};
};

inline std::size_t child_depth(std::size_t depth)
{
    if (depth == max_depth)
    {
        throw std::length_error{"archive exceeds the nesting limit"};
    }
    return depth + 1;
}
} // namespace minitensor::persistence::detail::binary
