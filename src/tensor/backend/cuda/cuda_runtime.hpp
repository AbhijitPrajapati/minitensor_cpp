#pragma once

#include <cstddef>
#include <span>

#include <minitensor/types.hpp>

#include "tensor/backend/device_runtime.hpp"
#include "tensor/storage/buffer.hpp"

namespace minitensor::detail::cuda
{
class CudaRuntime final : public DeviceRuntime
{
  public:
    explicit CudaRuntime(Device device = Device::cuda());
    [[nodiscard]] Device device() const noexcept override;
    [[nodiscard]] BufferRef allocate(std::size_t size_bytes) override;
    void copy_from_host(Buffer &destination, std::size_t destination_offset_bytes,
                        std::span<const std::byte> source) override;
    void copy_to_host(std::span<std::byte> destination, const Buffer &source, std::size_t source_offset_bytes) override;

  private:
    Device device_;
};
} // namespace minitensor::detail::cuda