#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include <minitensor/tensor.hpp>

namespace minitensor::persistence
{
class Record;
class Value;

template <typename T>
concept PersistenceScalar = std::same_as<T, bool> || std::same_as<T, std::int64_t> || std::same_as<T, std::uint64_t> ||
                            std::same_as<T, float> || std::same_as<T, double>;

template <typename T>
concept RecordValue = PersistenceScalar<T> || std::same_as<T, std::string> || std::same_as<T, Tensor> ||
                      std::same_as<T, Record> || std::same_as<T, std::vector<Value>>;

// Records and lists copy by value. Tensors copy only their handles.
class Value final
{
  public:
    using List = std::vector<Value>;

    // Kind mapping is coupled to enum index, so ordering must be preserved
    // relative to the Storage variant defined below.
    enum class Kind
    {
        Bool,
        Int64,
        UInt64,
        Float32,
        Float64,
        String,
        Tensor,
        Record,
        List
    };

    // Integral types are handled seperately for canonicalization.
    template <std::integral T>
    Value(T value)
    {
        static_assert(sizeof(T) <= sizeof(std::uint64_t));
        // Boolean values are preserved.
        if constexpr (std::same_as<T, bool>)
        {
            storage_ = value;
        }
        // Signed integers go to int64_t.
        else if constexpr (std::is_signed_v<T>)
        {
            storage_ = static_cast<std::int64_t>(value);
        }
        // Unsigned integers go to uint64_t.
        else
        {
            storage_ = static_cast<std::uint64_t>(value);
        }
    }

    // Float and double are the only accepted floating-point types.
    template <typename T>
        requires(std::same_as<T, float> || std::same_as<T, double>)
    Value(T value) : storage_{value}
    {
    }

    // Text is copied into an owning string. Null character pointers are rejected.
    Value(std::string value);
    Value(std::string_view value);
    Value(const char *value);
    Value(std::nullptr_t) = delete;

    Value(Tensor value);
    Value(Record value);
    Value(List value);

    Value(const Value &other);
    Value(Value &&) noexcept = default;
    Value &operator=(Value &&) noexcept = default;
    ~Value() = default;
    Value &operator=(const Value &other);

    [[nodiscard]] Kind kind() const noexcept;

    template <RecordValue T>
    [[nodiscard]] bool is() const noexcept
    {
        if constexpr (std::same_as<T, Record>)
        {
            return std::holds_alternative<std::unique_ptr<Record>>(storage_);
        }
        else
        {
            return std::holds_alternative<T>(storage_);
        }
    }

    // No implicit conversions are performed when reading values.
    template <RecordValue T>
    [[nodiscard]] const T &get() const
    {
        if constexpr (std::same_as<T, Record>)
        {
            const auto *value = std::get_if<std::unique_ptr<Record>>(&storage_);
            if (value != nullptr && *value)
            {
                return **value;
            }
        }
        else
        {
            if (const auto *value = std::get_if<T>(&storage_))
            {
                return *value;
            }
        }
        throw std::logic_error{"record value type does not match requested type"};
    }

  private:
    // Unique pointer indirection makes records recursive without sharing mutable record structure.
    using Storage = std::variant<bool, std::int64_t, std::uint64_t, float, double, std::string, Tensor,
                                 std::unique_ptr<Record>, List>;
    Storage storage_;
};

// Names are literal strings, including dots. Insertion never overwrites an entry.
// Inspection and copying do not evaluate or detach tensor values.
class Record final
{
  public:
    // Ordered map based on field names.
    using Fields = std::map<std::string, Value, std::less<>>;
    using const_iterator = Fields::const_iterator;

    Record() = default;

    // Throws invalid_argument on duplicate names.
    void add(std::string_view name, Value value);

    [[nodiscard]] bool contains(std::string_view name) const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    // Typed reads throw logic_error on a mismatch.
    [[nodiscard]] const Value &get(std::string_view name) const;

    // Const iteration follows lexicographic field-name order.
    [[nodiscard]] const_iterator begin() const noexcept;
    [[nodiscard]] const_iterator end() const noexcept;

  private:
    Fields fields_;
};

} // namespace minitensor::persistence
