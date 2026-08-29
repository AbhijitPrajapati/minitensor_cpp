#pragma once

#include <cstddef>
#include <memory>

namespace minitensor
{
    class Device;
}

namespace minitensor::detail
{
    class Buffer
    {
    public:
        Buffer(const Buffer &) = delete;
        Buffer &operator=(const Buffer &) = delete;
        virtual ~Buffer() = default;

        [[nodiscard]] virtual Device device() const noexcept = 0;
        [[nodiscard]] virtual std::size_t size_bytes() const noexcept = 0;

    protected:
        Buffer() = default;
    };

    using BufferRef = std::shared_ptr<Buffer>;
}
