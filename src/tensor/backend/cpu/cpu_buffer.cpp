#include "cpu_buffer.hpp"

#include <new>
#include <stdexcept>

namespace minitensor::detail::cpu
{
    CpuBuffer::CpuBuffer(std::size_t size_bytes, Device device) : size_bytes_(size_bytes), device_(device)
    {
        if (device_.type() != DeviceType::Cpu)
        {
            throw std::invalid_argument{"cpu device is required"};
        }

        if (size_bytes_ != 0)
        {
            data_ = static_cast<std::byte *>(::operator new(size_bytes_, std::align_val_t{alignment}));
        }
    }

    CpuBuffer::~CpuBuffer()
    {
        if (data_)
        {
            ::operator delete(data_, std::align_val_t{alignment});
        }
    }

    Device CpuBuffer::device() const noexcept
    {
        return device_;
    }

    std::size_t CpuBuffer::size_bytes() const noexcept
    {
        return size_bytes_;
    }

    std::byte *CpuBuffer::data() noexcept
    {
        return data_;
    }

    const std::byte *CpuBuffer::data() const noexcept
    {
        return data_;
    }
}