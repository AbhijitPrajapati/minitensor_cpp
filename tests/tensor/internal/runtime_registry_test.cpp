#include <minitensor/types.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

#include "tensor/backend/device_runtime.hpp"
#include "tensor/execution/runtime_registry.hpp"

#include "../support/test.hpp"
#include "../support/test_runtime.hpp"

namespace minitensor::test
{
    void run_runtime_registry_test()
    {
        using detail::DeviceRuntime;
        using detail::RuntimeRegistry;

        RuntimeRegistry registry;
        const Device primary_device = Device::cpu(3);
        expect(!registry.contains(primary_device), "a new registry contains no runtime");
        expect_throws<std::runtime_error>(
            [&registry, primary_device]
            {
                (void)registry.get(primary_device);
            },
            "getting an unregistered device runtime throws");

        expect_throws<std::invalid_argument>(
            [&registry]
            {
                registry.register_runtime(std::unique_ptr<DeviceRuntime>{});
            },
            "registering an empty runtime throws");
        expect(!registry.contains(Device::cpu()),
               "rejecting an empty runtime leaves the registry unchanged");

        auto primary = std::make_unique<TestRuntime>(primary_device);
        TestRuntime *const primary_address = primary.get();
        registry.register_runtime(std::move(primary));
        expect(primary == nullptr, "runtime registration takes ownership of the runtime");
        expect(registry.contains(primary_device), "a registered runtime is available by device");
        expect(&registry.get(primary_device) == primary_address,
               "get returns the runtime registered for the requested device");

        const RuntimeRegistry &const_registry = registry;
        expect(&const_registry.get(primary_device) == primary_address,
               "runtime lookup is available through a const registry");

        auto duplicate = std::make_unique<TestRuntime>(primary_device);
        expect_throws<std::logic_error>(
            [&registry, &duplicate]
            {
                registry.register_runtime(std::move(duplicate));
            },
            "registering a duplicate device runtime throws");
        expect(&registry.get(primary_device) == primary_address,
               "duplicate registration does not replace the original runtime");

        auto other = std::make_unique<TestRuntime>(Device::cpu());
        TestRuntime *const other_address = other.get();
        registry.register_runtime(std::move(other));
        expect(&registry.get(Device::cpu()) == other_address,
               "device indices distinguish runtime registry entries");

        bool owned_runtime_destroyed = false;
        {
            RuntimeRegistry owning_registry;
            owning_registry.register_runtime(
                std::make_unique<TestRuntime>(Device::cpu(), &owned_runtime_destroyed));
            expect(!owned_runtime_destroyed, "the registry keeps its registered runtime alive");
        }
        expect(owned_runtime_destroyed, "destroying the registry destroys its owned runtimes");
    }
}
