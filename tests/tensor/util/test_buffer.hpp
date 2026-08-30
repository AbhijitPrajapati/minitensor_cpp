#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include <minitensor/types.hpp>

#include "storage/buffer.hpp"

namespace minitensor::test
{

    class TestBuffer final : public detail::Buffer
    {
    public:
        TestBuffer(Device device, std::size_t size_bytes) : device_{device}, size_bytes_{size_bytes} {}

        [[nodiscard]] Device device() const noexcept override
        {
            return device_;
        }

        [[nodiscard]] std::size_t size_bytes() const noexcept override
        {
            return size_bytes_;
        }

    private:
        Device device_;
        std::size_t size_bytes_;
    };

    inline detail::BufferRef make_test_buffer(std::size_t size_bytes, Device device = Device::cpu())
    {
        return std::make_shared<TestBuffer>(device, size_bytes);
    }

}