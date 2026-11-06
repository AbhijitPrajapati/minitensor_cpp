#pragma once

#include <span>

#include <minitensor/nn/optimizers/adamw.hpp>
#include <minitensor/persistence/record.hpp>

#include "minitensor/persistence/detail/optimizer_persistence.hpp"
#include "minitensor/persistence/detail/parameter_binding.hpp"

namespace minitensor::persistence::detail
{
template <>
struct OptimizerPersistence<nn::AdamW> final
{
    [[nodiscard]] static Record capture_config(const nn::AdamW &optimizer);
    [[nodiscard]] static nn::AdamW restore_config(const Record &record);

    // Captures handles without evaluating or detaching moment tensors.
    [[nodiscard]] static Record capture_state(const nn::AdamW::State &state, const nn::ParameterMap &parameters,
                                              const ParameterBinding &binding);

    [[nodiscard]] static nn::AdamW::State restore_state(const Record &record, const nn::ParameterMap &parameters,
                                                        std::span<const nn::ParameterId> parameter_ids);
};
} // namespace minitensor::persistence::detail
