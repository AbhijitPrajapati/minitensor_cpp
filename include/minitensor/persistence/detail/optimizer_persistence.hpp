#pragma once

namespace minitensor::persistence::detail
{
// Specializations provide capture_config, restore_config, capture_state, and
// restore_state. Only the state operations use the parameter binding.
// Specializations must be visible to generic callers.
template <typename Optimizer>
struct OptimizerPersistence;
} // namespace minitensor::persistence::detail
