#pragma once

#include <cstdint>

#include <minitensor/tensor.hpp>

namespace minitensor::nn
{
class ParameterId final
{
  public:
    using value_type = std::uint64_t;

    ParameterId() = delete;

    [[nodiscard]] constexpr value_type value() const noexcept
    {
        return value_;
    }

    friend constexpr bool operator==(const ParameterId &, const ParameterId &) = default;

  private:
    explicit constexpr ParameterId(value_type value) noexcept : value_(value)
    {
    }

    // This friendship ensures that only Parameters can create IDs
    friend class Parameter;

    value_type value_;
};

struct ParameterMetadata
{
    bool trainable{true};
    friend bool operator==(const ParameterMetadata &, const ParameterMetadata &) = default;
};

class Parameter final
{
  public:
    // The public constructor always creates a fresh ID.
    explicit Parameter(Tensor value, ParameterMetadata metadata = {});
    Parameter(const Parameter &) noexcept = default;
    Parameter(Parameter &&) noexcept = default;
    Parameter &operator=(const Parameter &) noexcept = default;
    Parameter &operator=(Parameter &&) noexcept = default;

    [[nodiscard]] ParameterId id() const noexcept;
    [[nodiscard]] const Tensor &value() const noexcept;
    [[nodiscard]] const ParameterMetadata &metadata() const noexcept;

    // Parameters are constant by nature. Changing the value or metadata
    // produces a new Parameter.
    [[nodiscard]] Parameter with_value(Tensor value) const;
    [[nodiscard]] Parameter with_metadata(ParameterMetadata metadata) const;

  private:
    // A supplied ID must only be allowed privately.
    Parameter(ParameterId id, Tensor value, ParameterMetadata metadata);
    [[nodiscard]] static ParameterId make_id();

    ParameterId id_;
    Tensor value_;
    ParameterMetadata metadata_;
};
} // namespace minitensor::nn

namespace std
{
template <>
struct hash<minitensor::nn::ParameterId>
{
    [[nodiscard]] size_t operator()(const minitensor::nn::ParameterId &id) const noexcept
    {
        return hash<minitensor::nn::ParameterId::value_type>{}(id.value());
    }
};
} // namespace std