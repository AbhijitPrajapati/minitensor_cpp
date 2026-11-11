#pragma once

#include <cstddef>
#include <stdexcept>
#include <string_view>

#include <minitensor/nn/optimizers/sgd.hpp>
#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/parameters/parameter_id.hpp>
#include <minitensor/nn/persistence/detail/optimizer_persistence.hpp>
#include <minitensor/nn/persistence/detail/parameter_binding.hpp>
#include <minitensor/persistence/record.hpp>

namespace minitensor::nn::detail
{
template <>
struct OptimizerPersistence<SGD>
{
    static constexpr std::string_view payload_name{"sgd"};

    class Capture
    {
      public:
        Capture(const SGD &optimizer, const SGD::State &) : optimizer_{optimizer}
        {
        }

        void add_parameter(std::size_t, ParameterId, const ParameterState &)
        {
        }

        [[nodiscard]] persistence::Record finish() &&
        {
            persistence::Record record;
            record.add("learning_rate", optimizer_.learning_rate());
            return record;
        }

      private:
        const SGD &optimizer_;
    };

    static void validate_payload(const persistence::Record &record)
    {
        if (!record.contains({"learning_rate"}, true))
        {
            throw std::invalid_argument{"SGD checkpoint does not contain the expected fields"};
        }
    }

    [[nodiscard]] static SGD restore_optimizer(const persistence::Record &record)
    {
        return SGD{record.get("learning_rate").get<float>()};
    }

    [[nodiscard]] static SGD::State restore_state(const persistence::Record &, const ParameterBinding &,
                                                  const ParameterMap &)
    {
        return {};
    }
};
} // namespace minitensor::nn::detail
