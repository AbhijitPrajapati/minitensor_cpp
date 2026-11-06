#pragma once

#include <minitensor/persistence/detail/optimizer_persistence.hpp>
#include <minitensor/persistence/detail/optimizers/adamw.hpp>
#include <minitensor/persistence/record.hpp>

namespace minitensor::persistence
{
template <typename Optimizer>
[[nodiscard]] Record capture_optimizer_config(const Optimizer &optimizer)
{
    return detail::OptimizerPersistence<Optimizer>::capture_config(optimizer);
}

template <typename Optimizer>
[[nodiscard]] Optimizer restore_optimizer_config(const Record &record)
{
    return detail::OptimizerPersistence<Optimizer>::restore_config(record);
}
} // namespace minitensor::persistence
