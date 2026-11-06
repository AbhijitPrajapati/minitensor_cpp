#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>

#include <minitensor/nn/optimizers/adamw.hpp>

namespace minitensor::nn::detail
{
struct AdamWStateAccess final
{
    template <typename Function>
    static void for_each_moment(const AdamW::State &state, Function &&function)
    {
        for (const auto &[id, moment] : state.moments_)
        {
            std::invoke(function, id, moment.first, moment.second);
        }
    }

    struct MomentEntry
    {
        ParameterId id;
        Tensor first;
        Tensor second;
    };

    // The caller validates moment compatibility with the corresponding parameters.
    [[nodiscard]] static AdamW::State make(std::uint64_t step, std::span<const MomentEntry> moments)
    {
        AdamW::State state;
        state.step_ = step;
        state.moments_.reserve(moments.size());
        for (const MomentEntry &moment : moments)
        {
            if (!state.moments_.emplace(moment.id, AdamW::State::Moment{moment.first, moment.second}).second)
            {
                throw std::invalid_argument{"AdamW state contains duplicate parameter moments"};
            }
        }
        return state;
    }
};
} // namespace minitensor::nn::detail
