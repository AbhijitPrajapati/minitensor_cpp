#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <minitensor/nn/optimizers/adamw.hpp>
#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/persistence/detail/optimizer_persistence.hpp>
#include <minitensor/nn/persistence/detail/snapshot_binding.hpp>
#include <minitensor/persistence/record.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::nn::detail
{
class AdamWStateCodec
{
  public:
    explicit AdamWStateCodec(const AdamW::State &state) : state_{state}
    {
        moments_.reserve(state_.moments_.size());
    }

    void add_parameter(std::size_t index, ParameterId id, const ParameterState &parameter)
    {
        const auto moment = state_.moments_.find(id);
        if (!parameter.metadata.trainable)
        {
            if (moment != state_.moments_.end())
            {
                throw std::invalid_argument{"AdamW moments reference a frozen parameter"};
            }
            return;
        }
        if (moment == state_.moments_.end())
        {
            throw std::invalid_argument{"AdamW state is missing parameter moments"};
        }

        const auto &[first, second] = moment->second;
        validate_moments(parameter.value, first, second);

        persistence::Record entry;
        entry.add("parameter", static_cast<std::uint64_t>(index));
        entry.add("first", first);
        entry.add("second", second);
        moments_.emplace_back(std::move(entry));
    }

    void finish(persistence::Record &record) &&
    {
        if (moments_.size() != state_.moments_.size())
        {
            throw std::invalid_argument{"AdamW state contains unexpected parameter moments"};
        }
        record.add("step", state_.step());
        record.add("moments", std::move(moments_));
    }

    [[nodiscard]] static AdamW::State restore(const persistence::Record &record, const ParameterBinding &binding,
                                              const ParameterMap &parameters)
    {
        AdamW::State state;

        const auto step = record.get("step").get<std::uint64_t>();
        if (step == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::invalid_argument{"AdamW state cannot resume"};
        }
        state.step_ = step;

        const auto &moments = record.get("moments").get<persistence::Value::List>();
        state.moments_.reserve(std::min(moments.size(), parameters.size()));
        for (const auto &saved : moments)
        {
            const auto &entry = saved.get<persistence::Record>();
            if (!entry.contains({"parameter", "first", "second"}, true))
            {
                throw std::invalid_argument{"AdamW moment record does not contain the expected fields"};
            }
            const auto saved_index = entry.get("parameter").get<std::uint64_t>();
            if (saved_index >= binding.id_by_index.size())
            {
                throw std::out_of_range{"AdamW parameter reference is out of range"};
            }
            const ParameterId id = *binding.id_by_index[static_cast<std::size_t>(saved_index)];
            const ParameterState &parameter = parameters.get(id);
            if (!parameter.metadata.trainable)
            {
                throw std::invalid_argument{"AdamW moments reference a frozen parameter"};
            }

            const auto &first = entry.get("first").get<Tensor>();
            const auto &second = entry.get("second").get<Tensor>();
            validate_moments(parameter.value, first, second);
            if (!state.moments_.emplace(id, AdamW::State::Moment{first, second}).second)
            {
                throw std::invalid_argument{"AdamW state contains duplicate parameter moments"};
            }
        }

        for (const auto &[id, parameter] : parameters)
        {
            if (parameter.metadata.trainable && !state.moments_.contains(id))
            {
                throw std::invalid_argument{"AdamW state is missing parameter moments"};
            }
        }
        return state;
    }

  private:
    static void validate_moments(const Tensor &value, const Tensor &first, const Tensor &second)
    {
        if (!is_floating_point(value.dtype()))
        {
            throw std::invalid_argument{"AdamW requires floating-point trainable parameters"};
        }
        if (first.shape() != value.shape() || first.dtype() != value.dtype() || first.device() != value.device() ||
            second.shape() != value.shape() || second.dtype() != value.dtype() || second.device() != value.device())
        {
            throw std::invalid_argument{"AdamW moments do not match parameter specification"};
        }
    }

    // Only used for the capture accumulator.
    const AdamW::State &state_;
    persistence::Value::List moments_;
};

template <>
struct OptimizerPersistence<AdamW>
{
    static constexpr std::string_view payload_name{"adamw"};

    class Capture
    {
      public:
        Capture(const AdamW &optimizer, const AdamW::State &state) : optimizer_{optimizer}, state_capture_{state}
        {
        }

        void add_parameter(std::size_t index, ParameterId id, const ParameterState &parameter)
        {
            state_capture_.add_parameter(index, id, parameter);
        }

        [[nodiscard]] persistence::Record finish() &&
        {
            persistence::Record record;
            record.add("learning_rate", optimizer_.learning_rate());
            record.add("weight_decay", optimizer_.weight_decay());
            record.add("beta1", optimizer_.beta1());
            record.add("beta2", optimizer_.beta2());
            record.add("epsilon", optimizer_.epsilon());
            std::move(state_capture_).finish(record);
            return record;
        }

      private:
        const AdamW &optimizer_;
        AdamWStateCodec state_capture_;
    };

    static void validate_payload(const persistence::Record &record)
    {
        if (!record.contains({"learning_rate", "weight_decay", "beta1", "beta2", "epsilon", "step", "moments"}, true))
        {
            throw std::invalid_argument{"AdamW checkpoint does not contain the expected fields"};
        }
    }

    [[nodiscard]] static AdamW restore_optimizer(const persistence::Record &record)
    {
        return AdamW{record.get("learning_rate").get<float>(), record.get("weight_decay").get<float>(),
                     record.get("beta1").get<float>(), record.get("beta2").get<float>(),
                     record.get("epsilon").get<float>()};
    }

    [[nodiscard]] static AdamW::State restore_state(const persistence::Record &record, const ParameterBinding &binding,
                                                    const ParameterMap &parameters)
    {
        return AdamWStateCodec::restore(record, binding, parameters);
    }
};
} // namespace minitensor::nn::detail
