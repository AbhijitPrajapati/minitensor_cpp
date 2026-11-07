#include <minitensor/nn/optimizers/sgd.hpp>

#include <cmath>
#include <stdexcept>

#include <minitensor/persistence/record.hpp>

namespace minitensor::nn
{
SGD::SGD(float learning_rate) : learning_rate_(learning_rate)
{
    if (!std::isfinite(learning_rate_) || learning_rate_ < 0.0F)
    {
        throw std::invalid_argument{"SGD learning rate must be finite and non-negative"};
    }
}

float SGD::learning_rate() const noexcept
{
    return learning_rate_;
}

SGD::State leafify_optimizer_state(SGD::State state) noexcept
{
    return state;
}

constexpr persistence::detail::RecordHeaderSchema sgd_config_schema{"sgd_config", 1};

persistence::Record capture_optimizer_configuration(const SGD &optimizer)
{
    persistence::Record record = persistence::detail::make_object_record(sgd_config_schema);
    record.add("learning_rate", optimizer.learning_rate());
    return record;
}
SGD restore_sgd_configuration(const persistence::Record &record)
{
    persistence::detail::validate_object_record(record, sgd_config_schema, {"learning_rate"});
    return SGD{record.get("learning_rate").get<float>()};
}
} // namespace minitensor::nn
