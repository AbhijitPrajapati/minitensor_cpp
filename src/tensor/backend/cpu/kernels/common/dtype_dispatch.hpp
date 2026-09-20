#pragma once

#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <minitensor/types.hpp>

namespace minitensor::detail::cpu
{
    template <typename Function>
    decltype(auto) dispatch_dtype(DType dtype, Function &&function)
    {
        switch (dtype)
        {
        case DType::Float32:
            return std::invoke(std::forward<Function>(function), std::type_identity<float>{});
        }
        throw std::logic_error{"unsupported CPU dtype"};
    }
}
