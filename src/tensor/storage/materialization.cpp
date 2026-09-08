#include "materialization.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <cstdint>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/core/checked_arithmetic.hpp"

namespace minitensor::detail
{
    Materialization::Materialization(BufferRef buffer, Layout layout) : buffer_(std::move(buffer)), layout_(std::move(layout))
    {
        if (!buffer_)
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

    void Materialization::validate(const TensorSpec &spec) const
    {
        if (spec.device != buffer_->device())
        {
            throw std::invalid_argument{"buffer device does not match tensor specification"};
        }

        if (layout_.rank() != spec.shape.rank())
        {
            throw std::invalid_argument{"layout rank does not match tensor rank"};
        }

        if (spec.shape.numel() == 0)
        {
            return;
        }

        Layout::offset_type min_offset = layout_.offset();
        Layout::offset_type max_offset = layout_.offset();

        for (std::size_t axis = 0; axis < spec.shape.rank(); ++axis)
        {
            const auto dist = static_cast<Layout::offset_type>(spec.shape[axis] - 1);
            const Layout::offset_type contribution = checked_multiply(layout_.stride(axis), dist);

            if (contribution < 0)
            {
                min_offset = checked_add(min_offset, contribution);
            }
            else
            {
                max_offset = checked_add(max_offset, contribution);
            }
        }

        if (min_offset < 0)
        {
            throw std::invalid_argument{"materialization reaches before the beginning of its buffer"};
        }

        const std::size_t element_size = dtype_size(spec.dtype);
        const std::size_t buffer_capacity = buffer_->size_bytes() / element_size;

        const auto u_max_offset = static_cast<std::uintmax_t>(max_offset);
        const auto u_buffer_capacity = static_cast<std::uintmax_t>(buffer_capacity);

        if (u_max_offset >= u_buffer_capacity)
        {
            throw std::invalid_argument{"materialization reaches beyond the end of its buffer"};
        }
    }
}
