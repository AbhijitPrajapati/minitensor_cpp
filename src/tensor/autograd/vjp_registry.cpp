#include "vjp_registry.hpp"

#include <stdexcept>
#include <utility>

#include "rules/registrations.hpp"

namespace minitensor::detail
{
void VjpRegistry::register_erased(PrimitiveId id, VjpRule rule)
{
    auto [iterator, inserted] = rules_.try_emplace(id, std::move(rule));
    if (!inserted)
    {
        throw std::logic_error{"a VJP rule is already registered for this primitive"};
    }
}

const VjpRule &VjpRegistry::get(PrimitiveId id) const
{
    const auto iterator = rules_.find(id);
    if (iterator == rules_.end())
    {
        throw std::runtime_error{"no VJP rule is registered for this primitive"};
    }
    return iterator->second;
}

bool VjpRegistry::contains(PrimitiveId id) const
{
    return rules_.contains(id);
}

const VjpRegistry &vjp_registry()
{
    static const VjpRegistry registry = [] {
        VjpRegistry result;
        register_elementwise_vjps(result);
        register_linalg_vjps(result);
        register_manipulation_vjps(result);
        register_reduction_vjps(result);
        return result;
    }();
    return registry;
}
} // namespace minitensor::detail
