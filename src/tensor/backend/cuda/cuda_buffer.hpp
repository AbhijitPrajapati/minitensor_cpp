#pragma once

#include <cstddef>

#include <minitensor/types.hpp>

#include "tensor/storage/buffer.hpp"

namespace minitensor::detail::cuda
{
class CudaBuffer final : public Buffer
{
  public:
    explicit CudaBuffer(std::size_t size_bytes, Device device = Device::cuda());
    ~CudaBuffer() override;
    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    [[nodiscard]] Device device() const noexcept override;
    [[nodiscard]] std::size_t size_bytes() const noexcept override;

    // Returns a device pointer.
    [[nodiscard]] std::byte *data() noexcept;
    [[nodiscard]] const std::byte *data() const noexcept;

  private:
    Device device_;
    std::size_t size_bytes_;
    std::byte *data_{nullptr};
};
} // namespace minitensor::detail::cuda