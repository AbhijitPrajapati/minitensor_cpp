#include <minitensor/persistence/record.hpp>

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

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
    return static_cast<Kind>(storage_.index());
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
} // namespace minitensor::persistence
