#pragma once

#include <cassert>
#include <concepts>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/cpu_buffer.hpp"
#include "tensor/dispatch/tensor_view.hpp"

namespace minitensor::detail::cpu
{
    template <typename T>
    struct ElementDType;

    template <>
    struct ElementDType<float> final
    {
        static constexpr DType value = DType::Float32;
    };

    template <typename T>
    concept CpuElement = requires
    {
        { ElementDType<T>::value } -> std::convertible_to<DType>;
    };

    template <CpuElement T>
    [[nodiscard]] const T *data(const TensorView &view)
    {
        assert(view.dtype() == ElementDType<T>::value);
        const auto &buffer = dynamic_cast<const CpuBuffer &>(view.buffer());
        return reinterpret_cast<const T *>(buffer.data());
    }

    template <CpuElement T>
    [[nodiscard]] T *data(const MutableTensorView &view)
    {
        assert(view.dtype() == ElementDType<T>::value);
        auto &buffer = dynamic_cast<CpuBuffer &>(view.buffer());
        return reinterpret_cast<T *>(buffer.data());
    }
}
