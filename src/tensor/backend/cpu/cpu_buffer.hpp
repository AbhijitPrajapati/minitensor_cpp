#pragma once

#include <cstdint>

#include <minitensor/types.hpp>

#include "tensor/storage/buffer.hpp"

namespace minitensor::detail::cpu
{
    class CpuBuffer final : public Buffer
    {
    public:
        static constexpr std::size_t alignment = 64;

        explicit CpuBuffer(std::size_t size_bytes, Device device = Device::cpu());
        ~CpuBuffer() override;
        CpuBuffer(const CpuBuffer &) = delete;
        CpuBuffer &operator=(const CpuBuffer &) = delete;

        [[nodiscard]] Device device() const noexcept override;
        [[nodiscard]] std::size_t size_bytes() const noexcept override;
        [[nodiscard]] std::byte *data() noexcept;
        [[nodiscard]] const std::byte *data() const noexcept;

    private:
        Device device_;
        std::size_t size_bytes_;
        std::byte *data_{nullptr};
    };
}