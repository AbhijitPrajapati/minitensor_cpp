#include <minitensor/nn/parameter.hpp>

#include <atomic>
#include <limits>
#include <stdexcept>
#include <utility>

#include <minitensor/tensor.hpp>

namespace minitensor::nn
{
ParameterId ParameterBuilder::make_id()
{
    using value_type = ParameterId::value_type;
    static std::atomic<value_type> next{1};
    value_type current = next.load(std::memory_order_relaxed);

    do
    {
        if (current == std::numeric_limits<value_type>::max())
        {
            throw std::overflow_error{"parameter id space exhausted"};
        }
    } while (!next.compare_exchange_weak(current, current + 1, std::memory_order_relaxed, std::memory_order_relaxed));

    return ParameterId{current};
}

ParameterMap::ParameterMap(ParameterMap::Map states) : states_(std::move(states))
{
}

bool ParameterMap::contains(ParameterId id) const
{
    return states_.contains(id);
}

const ParameterState &ParameterMap::get(ParameterId id) const
{
    return states_.at(id);
}

std::size_t ParameterMap::size() const noexcept
{
    return states_.size();
}

ParameterId ParameterBuilder::add(ParameterState state)
{
    ParameterId id = make_id();
    auto [it, inserted] = parameters_.emplace(id, std::move(state));

    if (!inserted)
    {
        throw std::logic_error{"duplicate parameter ID"};
    }

    return id;
}

} // namespace minitensor::nn
