#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <minitensor/nn/optimizers/rmsprop.hpp>
#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/parameters/parameter_id.hpp>
#include <minitensor/nn/persistence/detail/optimizer_persistence.hpp>
#include <minitensor/nn/persistence/detail/parameter_binding.hpp>
#include <minitensor/persistence/record.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::nn::detail
{
template <>
struct OptimizerPersistence<RMSProp>
{
    static constexpr std::string_view payload_name{"rmsprop"};

    class Capture
    {
      public:
        Capture(const RMSProp &optimizer, const RMSProp::State &state) : optimizer_{optimizer}, state_{state}
        {
            averages_.reserve(state_.moving_average_.size());
        }

        void add_parameter(std::size_t index, ParameterId id, const ParameterState &parameter)
        {
            const auto average = state_.moving_average_.find(id);
            if (!parameter.metadata.trainable)
            {
                if (average != state_.moving_average_.end())
                {
                    throw std::invalid_argument{"RMSProp averages reference a frozen parameter"};
                }
                return;
            }
            if (average == state_.moving_average_.end())
            {
                throw std::invalid_argument{"RMSProp state is missing a parameter average"};
            }
            validate_average(parameter.value, average->second);

            persistence::Record entry;
            entry.add("parameter", static_cast<std::uint64_t>(index));
            entry.add("average", average->second);
            averages_.emplace_back(std::move(entry));
        }
        [[nodiscard]] persistence::Record finish() &&
        {
            if (averages_.size() != state_.moving_average_.size())
            {
                throw std::invalid_argument{"RMSProp state contains unexpected parameter averages"};
            }
            persistence::Record record;
            record.add("learning_rate", optimizer_.learning_rate());
            record.add("alpha", optimizer_.alpha());
            record.add("epsilon", optimizer_.epsilon());
            record.add("moving_averages", std::move(averages_));
            return record;
        }

      private:
        const RMSProp &optimizer_;
        const RMSProp::State &state_;
        persistence::Value::List averages_;
    };

    static void validate_payload(const persistence::Record &record)
    {
        if (!record.contains({"learning_rate", "alpha", "epsilon", "moving_averages"}, true))
        {
            throw std::invalid_argument{"RMSProp checkpoint does not contain the expected fields"};
        }
    }

    [[nodiscard]] static RMSProp restore_optimizer(const persistence::Record &record)
    {
        return RMSProp{record.get("learning_rate").get<float>(), record.get("alpha").get<float>(),
                       record.get("epsilon").get<float>()};
    }

    [[nodiscard]] static RMSProp::State restore_state(const persistence::Record &record,
                                                      const ParameterBinding &binding, const ParameterMap &parameters)
    {
        RMSProp::State state;
        const auto &averages = record.get("moving_averages").get<persistence::Value::List>();
        state.moving_average_.reserve(std::min(averages.size(), parameters.size()));
        for (const auto &saved : averages)
        {
            const auto &entry = saved.get<persistence::Record>();
            if (!entry.contains({"parameter", "average"}, true))
            {
                throw std::invalid_argument{"RMSProp average record does not contain the expected fields"};
            }
            const auto saved_index = entry.get("parameter").get<std::uint64_t>();
            if (saved_index >= binding.id_by_index.size())
            {
                throw std::out_of_range{"RMSProp parameter reference is out of range"};
            }
            const ParameterId id = *binding.id_by_index[static_cast<std::size_t>(saved_index)];
            const ParameterState &parameter = parameters.get(id);
            if (!parameter.metadata.trainable)
            {
                throw std::invalid_argument{"RMSProp averages reference a frozen parameter"};
            }
            const auto &average = entry.get("average").get<Tensor>();
            validate_average(parameter.value, average);
            if (!state.moving_average_.emplace(id, average).second)
            {
                throw std::invalid_argument{"RMSProp state contains duplicate parameter averages"};
            }
        }
        for (const auto &[id, parameter] : parameters)
        {
            if (parameter.metadata.trainable && !state.moving_average_.contains(id))
            {
                throw std::invalid_argument{"RMSProp state is missing a parameter average"};
            }
        }
        return state;
    }

  private:
    static void validate_average(const Tensor &value, const Tensor &average)
    {
        if (!is_floating_point(value.dtype()))
        {
            throw std::invalid_argument{"RMSProp requires floating-point trainable parameters"};
        }
        if (average.shape() != value.shape() || average.dtype() != value.dtype() || average.device() != value.device())
        {
            throw std::invalid_argument{"RMSProp average does not match parameter specification"};
        }
    }
};
} // namespace minitensor::nn::detail
