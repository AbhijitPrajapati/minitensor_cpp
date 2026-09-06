#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/backend/device_runtime.hpp"

#include "test_buffer.hpp"

namespace minitensor::test
{
    class TestRuntime final : public detail::DeviceRuntime
    {
    public:
        explicit TestRuntime(Device device = Device::cpu(), bool *destroyed = nullptr) noexcept
            : device_{device}, destroyed_{destroyed}
        {
            if (destroyed_ != nullptr)
            {
                *destroyed_ = false;
            }
        }

        ~TestRuntime() override
        {
            if (destroyed_ != nullptr)
            {
                *destroyed_ = true;
            }
        }

        [[nodiscard]] Device device() const noexcept override
        {
            return device_;
        }

        [[nodiscard]] detail::BufferRef allocate(std::size_t size_bytes) override
        {
            allocation_sizes_.push_back(size_bytes);
            return make_test_buffer(size_bytes, device_);
        }

        [[nodiscard]] std::span<const std::size_t> allocation_sizes() const noexcept
        {
            return allocation_sizes_;
        }

    private:
        Device device_;
        bool *destroyed_;
        std::vector<std::size_t> allocation_sizes_;
    };
}
