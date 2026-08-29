#pragma once

#include <memory>
#include "layout.hpp"
#include "buffer.hpp"

namespace minitensor::detail
{
    struct TensorSpec;
    class Materialization final
    {
    public:
        Materialization(BufferRef buffer, Layout layout);
        [[nodiscard]] const BufferRef &buffer_ref() const noexcept;
        [[nodiscard]] const Layout &layout() const noexcept;
        void validate(const TensorSpec &spec) const;

    private:
        Layout layout_;
        BufferRef buffer_;
    };
}