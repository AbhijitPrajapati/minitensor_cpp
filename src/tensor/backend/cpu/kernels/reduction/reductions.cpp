#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <span>
#include <type_traits>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/kernels/common/dtype_dispatch.hpp"
#include "tensor/backend/cpu/kernels/reduction/loops.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/primitives/max.hpp"
#include "tensor/primitives/mean.hpp"
#include "tensor/primitives/min.hpp"
#include "tensor/primitives/sum.hpp"

namespace minitensor::detail::cpu
{
    namespace
    {
        template <typename T>
        [[nodiscard]] constexpr T maximum_identity() noexcept
        {
            if constexpr (std::numeric_limits<T>::has_infinity)
            {
                return -std::numeric_limits<T>::infinity();
            }
            else
            {
                return std::numeric_limits<T>::lowest();
            }
        }

        template <typename T>
        [[nodiscard]] constexpr T minimum_identity() noexcept
        {
            if constexpr (std::numeric_limits<T>::has_infinity)
            {
                return std::numeric_limits<T>::infinity();
            }
            else
            {
                return std::numeric_limits<T>::max();
            }
        }

        template <typename T>
        [[nodiscard]] T combine_maximum(T accumulator, T value)
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                if (std::isnan(accumulator) || std::isnan(value))
                {
                    return std::numeric_limits<T>::quiet_NaN();
                }
                return std::fmax(accumulator, value);
            }
            else
            {
                return std::max(accumulator, value);
            }
        }

        template <typename T>
        [[nodiscard]] T combine_minimum(T accumulator, T value)
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                if (std::isnan(accumulator) || std::isnan(value))
                {
                    return std::numeric_limits<T>::quiet_NaN();
                }
                return std::fmin(accumulator, value);
            }
            else
            {
                return std::min(accumulator, value);
            }
        }

        void run_sum(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            assert(inputs.size() == 1);
            const auto &sum = dynamic_cast<const SumPrimitive &>(primitive);

            dispatch_dtype(
                output.dtype(),
                [&]<typename T>(std::type_identity<T>)
                {
                    reduce<T>(inputs.front(), output, sum.axes(), T{0},
                        [](T accumulator, T value)
                        {
                            return accumulator + value;
                        });
                });
        }

        void run_mean(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            assert(inputs.size() == 1);
            const auto &mean = dynamic_cast<const MeanPrimitive &>(primitive);

            dispatch_dtype(
                output.dtype(),
                [&]<typename T>(std::type_identity<T>)
                {
                    const T normalizer = static_cast<T>(
                        mean.reduction_size(inputs.front().shape()));
                    reduce<T>(inputs.front(), output, mean.axes(), T{0},
                        [](T accumulator, T value)
                        {
                            return accumulator + value;
                        },
                        [normalizer](T value)
                        {
                            return value / normalizer;
                        });
                });
        }

        void run_max(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            assert(inputs.size() == 1);
            const auto &maximum = dynamic_cast<const MaxPrimitive &>(primitive);

            dispatch_dtype(
                output.dtype(),
                [&]<typename T>(std::type_identity<T>)
                {
                    reduce<T>(inputs.front(), output, maximum.axes(),
                        maximum_identity<T>(),
                        [](T accumulator, T value)
                        {
                            return combine_maximum(accumulator, value);
                        });
                });
        }

        void run_min(DeviceRuntime &, const Primitive &primitive, std::span<const TensorView> inputs, MutableTensorView output)
        {
            assert(inputs.size() == 1);
            const auto &minimum = dynamic_cast<const MinPrimitive &>(primitive);

            dispatch_dtype(
                output.dtype(),
                [&]<typename T>(std::type_identity<T>)
                {
                    reduce<T>(inputs.front(), output, minimum.axes(),
                        minimum_identity<T>(),
                        [](T accumulator, T value)
                        {
                            return combine_minimum(accumulator, value);
                        });
                });
        }
    }

    void register_reduction_kernels(KernelRegistry &registry)
    {
        registry.register_kernel(KernelKey{typeid(SumPrimitive), DeviceType::Cpu, DType::Float32}, run_sum);
        registry.register_kernel(KernelKey{typeid(MeanPrimitive), DeviceType::Cpu, DType::Float32}, run_mean);
        registry.register_kernel(KernelKey{typeid(MaxPrimitive), DeviceType::Cpu, DType::Float32}, run_max);
        registry.register_kernel(KernelKey{typeid(MinPrimitive), DeviceType::Cpu, DType::Float32}, run_min);
    }
}
