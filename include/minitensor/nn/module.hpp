#pragma once

#include <utility>

#include <minitensor/nn/parameter.hpp>
#include <minitensor/random.hpp>

namespace minitensor::nn
{
template <typename Module>
[[nodiscard]] ParameterSnapshot<typename Module::Parameters> initialize(const Module &module, RandomKey key,
                                                                        TensorOptions options = {})
{
    ParameterBuilder builder;
    auto tree = module.initialize(builder, key, options);
    return std::move(builder).finish(std::move(tree));
}

// Nested modules receive the shared map and their own child tree. At the root,
// the snapshot keeps those two pieces together and validates their agreement.
template <typename Module, typename Input>
[[nodiscard]] auto forward(const Module &module, const ParameterSnapshot<typename Module::Parameters> &snapshot,
                           const Input &input)
{
    return module(snapshot.parameters(), snapshot.tree(), input);
}
} // namespace minitensor::nn
