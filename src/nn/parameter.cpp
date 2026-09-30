#include <minitensor/nn/parameter.hpp>

#include <atomic>
#include <limits>
#include <stdexcept>
#include <utility>

#include <minitensor/tensor.hpp>

namespace minitensor::nn
{
	Parameter::Parameter(ParameterId id, Tensor value, ParameterMetadata metadata)
		: id_(id), value_(std::move(value)), metadata_(metadata)
	{}

	ParameterId Parameter::id() const noexcept
	{
		return id_;
	}

	const Tensor& Parameter::value() const noexcept
	{
		return value_;
	}

	const ParameterMetadata& Parameter::metadata() const noexcept
	{
		return metadata_;
	}

	ParameterId Parameter::make_id()
	{
		using value_type = ParameterId::value_type;
		static std::atomic<value_type> next{ 1 };
		value_type current = next.load(std::memory_order_relaxed);

		do
		{
			if (current == std::numeric_limits<value_type>::max())
			{
				throw std::overflow_error{ "parameter id space exhausted" };
			}
		} while (!next.compare_exchange_weak(current,
			current + 1,
			std::memory_order_relaxed,
			std::memory_order_relaxed));

		return ParameterId{ current };
	}

	Parameter::Parameter(Tensor value, ParameterMetadata metadata)
		: id_(make_id()), value_(std::move(value)), metadata_(metadata)
	{}

	Parameter Parameter::with_metadata(ParameterMetadata metadata) const
	{
		return Parameter{ id_, value_, metadata };
	}

	Parameter Parameter::with_value(Tensor value) const
	{
		if (value.shape() != value_.shape())
		{
			throw std::invalid_argument{"replacement value must have the same shape"};
		}

		if (value.dtype() != value_.dtype())
		{
			throw std::invalid_argument{ "replacement value must have the same dtype" };
		}

		if (value.device() != value_.device())
		{
			throw std::invalid_argument{ "replacement value must be on the same device" };
		}

		return Parameter{ id_, std::move(value), metadata_};
	}
}