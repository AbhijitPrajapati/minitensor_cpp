#pragma once

#include <cstddef>

#include <minitensor/nn/parameters/parameter.hpp>
#include <minitensor/nn/parameters/parameter_id.hpp>

namespace minitensor::nn::detail
{
// Specializations supply a capture accumulator and configuration/state restoration.
template <typename Optimizer>
struct OptimizerPersistence;

template <>
struct OptimizerPersistence<void>
{
    struct Capture
    {
        void add_parameter(std::size_t, ParameterId, const ParameterState &)
        {
        }
    };
};
} // namespace minitensor::nn::detail
