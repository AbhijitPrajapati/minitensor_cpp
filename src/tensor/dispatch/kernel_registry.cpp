#include <stdexcept>

#include "kernel_registry.hpp"

namespace minitensor::detail
{
    void KernelRegistry::register_kernel(KernelKey key, KernelFn kernel)
    {
        if (!kernel)
        {
            throw std::invalid_argument{"cannot register an empty kernel"};
        }
        auto [iterator, inserted] = kernels_.try_emplace(std::move(key), std::move(kernel));
        if (!inserted)
        {
            throw std::logic_error{"a kernel is already registered for this key"};
        }
    }
    const KernelFn &KernelRegistry::get(KernelKey &key) const
    {
        const auto iterator = kernels_.find(key);
        if (iterator == kernels_.end())
        {
            throw std::runtime_error{"no kernel is registered for this key"};
        }
        return iterator->second;
    }
    bool KernelRegistry::contains(KernelKey &key) const
    {
        return kernels_.contains(key);
    }
}