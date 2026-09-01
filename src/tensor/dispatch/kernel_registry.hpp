#pragma once

#include <functional>
#include <unordered_map>
#include <span>

#include "kernel_key.hpp"
#include "tensor_view.hpp"

namespace minitensor::detail
{
    class DeviceRuntime;
    class Primitive;

    using KernelFn = std::function<void(DeviceRuntime &, const Primitive &, std::span<const TensorView>, MutableTensorView)>;

    class KernelRegistry final
    {
    public:
        void register_kernel(KernelKey key, KernelFn kernel);
        [[nodiscard]] const KernelFn &get(KernelKey &key) const;
        [[nodiscard]] bool contains(KernelKey &key) const;

    private:
        std::unordered_map<KernelKey, KernelFn> kernels_;
    };
}