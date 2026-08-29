#include "materialization.hpp"
#include "layout.hpp"
#include "buffer.hpp"
#include <stdexcept>

namespace minitensor::detail
{
    // class Materialization final
    // {
    // public:
    //     Materialization(BufferRef buffer, Layout layout);
    //     [[nodiscard]] const BufferRef &buffer_ref() const noexcept;
    //     [[nodiscard]] const Layout &layout() const noexcept;

    // private:
    //     Layout layout_;
    //     BufferRef buffer;
    // };
    Materialization::Materialization(BufferRef buffer, Layout layout) : buffer_(std::move(buffer)), layout_(std::move(layout))
    {
        if (!buffer)
        {
            throw std::invalid_argument{"materialization requires a buffer"};
        }
    }

    const BufferRef &Materialization::buffer_ref() const noexcept
    {
        return buffer_;
    }

    const Layout &Materialization::layout() const noexcept
    {
        return layout_;
    }
}