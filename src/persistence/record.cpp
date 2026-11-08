#include <algorithm>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include <minitensor/persistence/record.hpp>

#include "binary.hpp"

namespace minitensor::persistence
{
Value::Value(std::string value) : storage_{std::move(value)}
{
}

Value::Value(std::string_view value) : storage_{std::string{value}}
{
}

Value::Value(const char *value)
    : Value{value != nullptr ? std::string{value} : throw std::invalid_argument{"null string value"}}
{
}

Value::Value(Tensor value) : storage_{std::move(value)}
{
}

Value::Value(Record value) : storage_{std::make_unique<Record>(std::move(value))}
{
}

Value::Value(List value) : storage_{std::move(value)}
{
}

// Use std::visit to obtain compile time typing for the storage
// of the other value.
Value::Value(const Value &other)
    : storage_{std::visit(
          [](const auto &value) -> Storage {
              using T = std::remove_cvref_t<decltype(value)>;
              if constexpr (std::same_as<T, std::unique_ptr<Record>>)
              {
                  return value ? std::make_unique<Record>(*value) : std::unique_ptr<Record>{};
              }
              else
              {
                  return value;
              }
          },
          other.storage_)}
{
}

Value &Value::operator=(const Value &other)
{
    if (this != &other)
    {
        Value copy{other};
        *this = std::move(copy);
    }
    return *this;
}

Value::Kind Value::kind() const noexcept
{
    return std::visit(
        []<typename T>(const T &) noexcept -> Kind {
            if constexpr (std::same_as<T, bool>)
                return Kind::Bool;
            else if constexpr (std::same_as<T, std::int64_t>)
                return Kind::Int64;
            else if constexpr (std::same_as<T, std::uint64_t>)
                return Kind::UInt64;
            else if constexpr (std::same_as<T, float>)
                return Kind::Float32;
            else if constexpr (std::same_as<T, double>)
                return Kind::Float64;
            else if constexpr (std::same_as<T, std::string>)
                return Kind::String;
            else if constexpr (std::same_as<T, Tensor>)
                return Kind::Tensor;
            else if constexpr (std::same_as<T, std::unique_ptr<Record>>)
                return Kind::Record;
            else
            {
                static_assert(std::same_as<T, List>, "Unhandled Value storage type");
                return Kind::List;
            }
        },
        storage_);
}

void Record::add(std::string_view name, Value value)
{
    if (!fields_.try_emplace(std::string{name}, std::move(value)).second)
    {
        throw std::invalid_argument{"duplicate record field name"};
    }
}

bool Record::contains(std::string_view name) const
{
    return fields_.contains(name);
}

bool Record::contains(std::initializer_list<std::string_view> names, bool exact) const
{
    if (exact)
    {
        if (fields_.size() != names.size())
        {
            return false;
        }

        return std::all_of(fields_.begin(), fields_.end(), [&](const auto &field) {
            return std::find(names.begin(), names.end(), std::string_view{field.first}) != names.end();
        });
    }

    return std::all_of(names.begin(), names.end(), [&](std::string_view name) { return contains(name); });
}
std::size_t Record::size() const noexcept
{
    return fields_.size();
}

bool Record::empty() const noexcept
{
    return fields_.empty();
}

const Value &Record::get(std::string_view name) const
{
    const auto position = fields_.find(name);
    if (position == fields_.end())
    {
        throw std::out_of_range{"record field not found"};
    }
    return position->second;
}

Record::const_iterator Record::begin() const noexcept
{
    return fields_.begin();
}

Record::const_iterator Record::end() const noexcept
{
    return fields_.end();
}

void Record::write(std::ostream &output) const
{
    detail::write_record(output, *this);
}

Record Record::read(std::istream &input, const ReadOptions &options)
{
    return detail::read_record(input, options);
}

namespace detail
{
Record make_object_record(RecordHeaderSchema schema)
{
    Record record;
    record.add("kind", schema.kind);
    record.add("version", schema.version);
    return record;
}

void validate_object_record(const Record &record, RecordHeaderSchema schema,
                            std::initializer_list<std::string_view> payload_fields)
{
    if (!record.contains({"kind", "version"}) || !record.contains(payload_fields))
    {
        throw std::invalid_argument{"record is missing required fields"};
    }

    // Validate exact payload fields.
    for (const auto &[name, value] : record)
    {
        if (name == "kind" || name == "version")
        {
            continue;
        }

        if (std::find(payload_fields.begin(), payload_fields.end(), std::string_view{name}) == payload_fields.end())
        {
            throw std::invalid_argument{"record contains an unexpected field"};
        }
    }

    if (record.get("kind").get<std::string>() != schema.kind)
    {
        throw std::invalid_argument{"unsupported record kind"};
    }
    if (record.get("version").get<std::uint64_t>() != schema.version)
    {
        throw std::invalid_argument{"unsupported record version"};
    }
}
} // namespace detail

} // namespace minitensor::persistence
