#pragma once

#include <concepts>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <minitensor/tensor.hpp>

#include "tensor/graph/primitive.hpp"

namespace minitensor::detail
{
struct VjpContext final
{
    std::span<const Tensor> inputs;
    const Tensor &output;
    const Tensor &output_cotangent;
};

using VjpContributions = std::vector<std::optional<Tensor>>;

using VjpRule = std::function<VjpContributions(const Primitive &primitive, const VjpContext &context)>;

template <typename PrimitiveType>
using TypedVjpRule = std::function<VjpContributions(const PrimitiveType &primitive, const VjpContext &context)>;

class VjpRegistry final
{
  public:
    template <typename PrimitiveType>
        requires std::derived_from<PrimitiveType, Primitive>
    void register_rule(TypedVjpRule<PrimitiveType> rule)
    {
        if (!rule)
        {
            throw std::invalid_argument{"cannot register an empty VJP rule"};
        }

        register_erased(PrimitiveType::ID,
                        [rule = std::move(rule)](const Primitive &primitive, const VjpContext &context) {
                            // We must cast to the concrete primitive type before calling the rule
                            // so that rules can access member functions of the concrete primitive.
                            const auto *typed = dynamic_cast<const PrimitiveType *>(&primitive);
                            if (typed == nullptr)
                            {
                                throw std::logic_error{"VJP registry entry does not match the primitive type"};
                            }
                            return rule(*typed, context);
                        });
    }

    [[nodiscard]] const VjpRule &get(PrimitiveId id) const;
    [[nodiscard]] bool contains(PrimitiveId id) const;

  private:
    // This function registers a type-erased VJP rule with an abstract Primitive.
    void register_erased(PrimitiveId id, VjpRule rule);

    std::unordered_map<PrimitiveId, VjpRule, PrimitiveIdHash> rules_;
};

// Singleton registry
[[nodiscard]] const VjpRegistry &vjp_registry();
} // namespace minitensor::detail
