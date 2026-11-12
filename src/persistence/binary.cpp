#include "binary.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <minitensor/data.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "binary_format.hpp"
#include "binary_io.hpp"

namespace minitensor::persistence::detail
{
namespace
{
using namespace binary;

// Empty tensors also need representable contiguous strides when reconstructed.
void validate_dimensions(std::span<const Extent> dimensions)
{
    std::uint64_t product = 1;
    for (Extent extent : dimensions)
    {
        if (extent < 0)
        {
            throw std::invalid_argument{"negative archive tensor extent"};
        }
        if (extent > 1)
        {
            const auto size = static_cast<std::uint64_t>(extent);
            if (product > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / size)
            {
                throw std::length_error{"archive tensor strides overflow"};
            }
            product *= size;
        }
    }
}

template <typename T>
void grow(std::vector<T> &values, std::size_t required, std::size_t maximum)
{
    if (required > values.capacity())
    {
        const auto capacity = std::min(maximum, std::max(required, 2 * values.capacity()));
        values.reserve(capacity);
    }
}

class Writer
{
  public:
    void record(const Record &record, std::size_t depth = 0)
    {
        if (record.size() > max_fields)
        {
            throw std::length_error{"record field count exceeds archive limit"};
        }
        budget_.record(record.size());
        number(static_cast<std::uint32_t>(record.size()));
        for (const auto &[name, value] : record)
        {
            text(name, max_text_bytes);
            write_value(value, depth);
        }
    }

    void finish(std::ostream &output)
    {
        emit_archive(output, payload_);
    }

  private:
    template <std::unsigned_integral T>
    void number(T value)
    {
        std::array<std::byte, sizeof(T)> bytes;
        store_le<T>(bytes, value);
        payload_.append(bytes);
    }

    void tag(Kind kind)
    {
        number(static_cast<std::uint8_t>(kind));
    }

    void text(std::string_view value, std::size_t limit)
    {
        if (value.size() > limit)
        {
            throw std::length_error{"text length exceeds archive limit"};
        }
        budget_.text(value.size());
        number(static_cast<std::uint32_t>(value.size()));
        payload_.append(std::as_bytes(std::span{value}));
    }

    template <typename T>
    void elements(const Tensor &tensor)
    {
        const auto values = to_vector<T>(tensor);
        const auto bytes = std::as_bytes(std::span{values});
        if constexpr (std::endian::native == std::endian::little)
        {
            payload_.append(bytes);
        }
        else
        {
            std::array<std::byte, frame_bytes> buffer;
            for (std::size_t offset = 0; offset < bytes.size();)
            {
                const auto count = std::min(buffer.size(), bytes.size() - offset);
                std::copy_n(bytes.data() + offset, count, buffer.data());
                for (std::size_t element = 0; element < count; element += sizeof(T))
                {
                    std::reverse(buffer.begin() + element, buffer.begin() + element + sizeof(T));
                }
                payload_.append(std::span{buffer}.first(count));
                offset += count;
            }
        }
    }

    void tensor(const Tensor &tensor)
    {
        TensorType dtype;
        std::size_t element_bytes;
        switch (tensor.dtype())
        {
        case DType::Float32:
            dtype = TensorType::Float32;
            element_bytes = 4;
            break;
        case DType::Int64:
            dtype = TensorType::Int64;
            element_bytes = 8;
            break;
        default:
            throw std::invalid_argument{"unsupported archive tensor dtype"};
        }
        if (tensor.rank() > max_tensor_rank)
        {
            throw std::length_error{"tensor rank exceeds archive limit "};
        }
        if (tensor.numel() > max_tensor_bytes / element_bytes)
        {
            throw std::length_error{"tensor element count exceeds archive limit "};
        }
        validate_dimensions(tensor.shape().dimensions());
        budget_.tensor(tensor.numel() * element_bytes, tensor.rank());
        number(static_cast<std::uint8_t>(dtype));
        number(static_cast<std::uint32_t>(tensor.rank()));
        for (Extent extent : tensor.shape())
        {
            number(static_cast<std::uint64_t>(extent));
        }
        switch (dtype)
        {
        case TensorType::Float32:
            elements<float>(tensor);
            return;
        case TensorType::Int64:
            elements<std::int64_t>(tensor);
            return;
        }
        throw std::invalid_argument{"unsupported archive tensor dtype"};
    }

    void write_value(const Value &value, std::size_t depth)
    {
        switch (value.kind())
        {
        case Value::Kind::Bool:
            tag(Kind::Bool);
            number<std::uint8_t>(value.get<bool>() ? 1 : 0);
            return;
        case Value::Kind::Int64:
            tag(Kind::Int64);
            number(std::bit_cast<std::uint64_t>(value.get<std::int64_t>()));
            return;
        case Value::Kind::UInt64:
            tag(Kind::UInt64);
            number(value.get<std::uint64_t>());
            return;
        case Value::Kind::Float32:
            tag(Kind::Float32);
            number(std::bit_cast<std::uint32_t>(value.get<float>()));
            return;
        case Value::Kind::Float64:
            tag(Kind::Float64);
            number(std::bit_cast<std::uint64_t>(value.get<double>()));
            return;
        case Value::Kind::String:
            tag(Kind::String);
            text(value.get<std::string>(), max_text_bytes);
            return;
        case Value::Kind::Tensor:
            tag(Kind::Tensor);
            tensor(value.get<Tensor>());
            return;
        case Value::Kind::Record:
            tag(Kind::Record);
            record(value.get<Record>(), child_depth(depth));
            return;
        case Value::Kind::List: {
            const auto next_depth = child_depth(depth);
            const auto &list = value.get<Value::List>();
            if (list.size() > max_list_values)
            {
                throw std::length_error{"list value count exceeds archive limit"};
            }
            budget_.list(list.size());
            tag(Kind::List);
            number(static_cast<std::uint32_t>(list.size()));
            for (const auto &element : list)
            {
                write_value(element, next_depth);
            }
            return;
        }
        }
        throw std::invalid_argument{"unsupported archive value kind"};
    }

    Spool payload_;
    Budget budget_;
};

class Reader
{
  public:
    Reader(ArchiveInput &input, Device device) : input_{input}, device_{device}
    {
    }

    [[nodiscard]] Record record(std::size_t depth = 0)
    {
        const auto count = size(max_fields, "record field count");
        input_.require(static_cast<std::uint64_t>(count) * 5);
        budget_.record(count);
        Record record;
        for (std::size_t index = 0; index < count; ++index)
        {
            const auto name = text();
            if (record.contains(name))
            {
                input_.fail("duplicate archive field name");
            }
            record.add(name, read_value(depth));
        }
        return record;
    }

  private:
    template <std::unsigned_integral T>
    [[nodiscard]] T number()
    {
        std::array<std::byte, sizeof(T)> bytes;
        input_.read(bytes);
        return load_le<T>(bytes);
    }

    [[nodiscard]] std::size_t size(std::size_t limit, std::string_view description)
    {
        const auto count = number<std::uint32_t>();
        if (!std::in_range<std::size_t>(count) || count > limit)
        {
            throw std::length_error{std::string{description} + " exceeds archive limit"};
        }
        return static_cast<std::size_t>(count);
    }

    [[nodiscard]] std::string text()
    {
        const auto count = size(max_text_bytes, "text length");
        input_.require(count);
        budget_.text(count);
        std::string value(count, '\0');
        input_.read(std::as_writable_bytes(std::span{value.data(), value.size()}));
        return value;
    }

    template <typename T>
    [[nodiscard]] Tensor elements(Shape shape)
    {
        const auto numel = shape.numel();
        if (numel > max_tensor_bytes / sizeof(T))
        {
            throw std::length_error{"tensor element count exceeds archive limit"};
        }
        input_.require(static_cast<std::uint64_t>(numel) * sizeof(T));
        budget_.tensor(numel * sizeof(T), shape.rank());
        // Grow only as payload arrives. A forged length never allocates the
        // entire tensor up front, even when a framed archive is truncated.
        std::vector<T> values;
        while (values.size() < numel)
        {
            const auto offset = values.size();
            const auto count = std::min(frame_bytes / sizeof(T), numel - offset);
            grow(values, offset + count, numel);
            values.resize(offset + count);
            auto bytes = std::as_writable_bytes(std::span{values}.subspan(offset, count));
            input_.read(bytes);
            if constexpr (std::endian::native == std::endian::big)
            {
                for (std::size_t element = 0; element < bytes.size(); element += sizeof(T))
                {
                    std::reverse(bytes.begin() + element, bytes.begin() + element + sizeof(T));
                }
            }
        }
        return from_data<T>(std::span<const T>{values}, std::move(shape), device_);
    }

    [[nodiscard]] Tensor tensor()
    {
        const auto dtype = static_cast<TensorType>(number<std::uint8_t>());
        if (dtype != TensorType::Float32 && dtype != TensorType::Int64)
        {
            input_.fail("unknown archive tensor dtype");
        }
        const auto rank = size(max_tensor_rank, "tensor rank");
        input_.require(static_cast<std::uint64_t>(rank) * 8);
        std::vector<Extent> dimensions;
        dimensions.reserve(rank);
        for (std::size_t axis = 0; axis < rank; ++axis)
        {
            const auto extent = number<std::uint64_t>();
            if (!std::in_range<Extent>(extent) || !std::in_range<std::size_t>(extent))
            {
                throw std::length_error{"archive tensor extent is too large"};
            }
            dimensions.push_back(static_cast<Extent>(extent));
        }
        validate_dimensions(dimensions);
        Shape shape{std::move(dimensions)};
        switch (dtype)
        {
        case TensorType::Float32:
            return elements<float>(std::move(shape));
        case TensorType::Int64:
            return elements<std::int64_t>(std::move(shape));
        }
        input_.fail("unknown archive tensor dtype");
    }

    [[nodiscard]] Value read_value(std::size_t depth)
    {
        const auto kind = static_cast<Kind>(number<std::uint8_t>());
        switch (kind)
        {
        case Kind::Bool: {
            const auto value = number<std::uint8_t>();
            if (value > 1)
            {
                input_.fail("invalid archive boolean");
            }
            return Value{value != 0};
        }
        case Kind::Int64:
            return Value{std::bit_cast<std::int64_t>(number<std::uint64_t>())};
        case Kind::UInt64:
            return Value{number<std::uint64_t>()};
        case Kind::Float32:
            return Value{std::bit_cast<float>(number<std::uint32_t>())};
        case Kind::Float64:
            return Value{std::bit_cast<double>(number<std::uint64_t>())};
        case Kind::String:
            return Value{text()};
        case Kind::Tensor:
            return Value{tensor()};
        case Kind::Record:
            return Value{record(child_depth(depth))};
        case Kind::List: {
            const auto next_depth = child_depth(depth);
            const auto count = size(max_list_values, "list value count");
            input_.require(count);
            budget_.list(count);
            Value::List values;
            for (std::size_t index = 0; index < count; ++index)
            {
                auto value = read_value(next_depth);
                grow(values, index + 1, count);
                values.emplace_back(std::move(value));
            }
            return Value{std::move(values)};
        }
        }
        input_.fail("unknown archive value kind");
    }

    ArchiveInput &input_;
    Device device_;
    Budget budget_;
};
} // namespace

void write_record(std::ostream &output, const Record &record)
{
    if (!output || output.rdbuf() == nullptr)
    {
        throw std::ios_base::failure{"archive destination is not writable"};
    }
    Writer writer;
    writer.record(record);
    writer.finish(output);
}

Record read_record(std::istream &input, const Record::ReadOptions &options, ArchiveBoundary boundary)
{
    binary::ArchiveInput archive{input};
    try
    {
        Reader reader{archive, options.device};
        auto record = reader.record();
        archive.finish(boundary);
        return record;
    }
    catch (const std::length_error &error)
    {
        throw std::length_error{std::string{error.what()} + " at payload byte " + std::to_string(archive.consumed())};
    }
}
} // namespace minitensor::persistence::detail
