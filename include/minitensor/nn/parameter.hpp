#pragma once

#include <cstdint>
#include <type_traits>

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

		friend constexpr bool operator==(const ParameterId&, const ParameterId&) = default;
	private:
		explicit constexpr ParameterId(value_type value) noexcept
			: value_(value)
		{}
		friend class Parameter; // only parameters can create ids
		value_type value_;
	};

	struct ParameterMetadata
	{
		bool trainable{ true };
		friend bool operator==(const ParameterMetadata&, const ParameterMetadata&) = default;
	};


	class Parameter final
	{
	public:
		// always allocates fresh id
		explicit Parameter(Tensor value, ParameterMetadata metadata = {});
		Parameter(const Parameter&) noexcept = default;
		Parameter(Parameter&&) noexcept = default;
		Parameter& operator=(const Parameter&) noexcept = default;
		Parameter& operator=(Parameter&&) noexcept = default;

		[[nodiscard]] ParameterId id() const noexcept;
		[[nodiscard]] const Tensor& value() const noexcept;
		[[nodiscard]] const ParameterMetadata& metadata() const noexcept;

		[[nodiscard]] Parameter with_value(Tensor value) const;
		[[nodiscard]] Parameter with_metadata(ParameterMetadata metadata) const;

	private:
		// always uses supplied id
		Parameter(ParameterId id, Tensor value, ParameterMetadata metadata);
		[[nodiscard]] static ParameterId make_id();

		ParameterId id_;
		Tensor value_;
		ParameterMetadata metadata_;
	};
}

namespace std
{
	template <>
	struct hash<minitensor::nn::ParameterId>
	{
		[[nodiscard]] size_t operator()(const minitensor::nn::ParameterId& id) const noexcept
		{
			return hash<minitensor::nn::ParameterId::value_type>{}(id.value());
		}
	};
}