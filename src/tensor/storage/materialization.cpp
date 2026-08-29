#include "materialization.hpp"
#include "layout.hpp"
#include "buffer.hpp"
#include "tensor/core/tensor_spec.hpp"
#include <stdexcept>
#include <limits>

namespace minitensor::detail
{

    namespace
    {
        using offset_type = Layout::offset_type;
        using stride_type = Layout::stride_type;

        constexpr auto min = std::numeric_limits<offset_type>::min();
        constexpr auto max = std::numeric_limits<offset_type>::max();

        offset_type checked_add(offset_type lhs, offset_type rhs)
        {

            if (rhs > 0 && lhs > max - rhs)
            {
                throw std::overflow_error{"layout offset overflow"};
            }
            if (rhs < 0 && lhs < min - rhs)
            {
                throw std::overflow_error{"layout offset underflow"};
            }
            return lhs + rhs;
        }

        offset_type checked_contribution(stride_type stride, Extent dist)
        {
            if (stride == 0 || dist == 0)
            {
                return 0;
            }

            if (stride > 0 && stride > max / dist)
            {
                throw std::overflow_error{"layout stride contribution overflow"};
            }
            if (stride < 0 && stride < min / dist)
            {
                throw std::overflow_error{"layout stride contribution underflow"};
            }
            return stride * dist;
        }
    }

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

        offset_type min_offset = layout_.offset();
        offset_type max_offset = layout_.offset();

        for (std::size_t axis = 0; axis < spec.shape.rank(); ++axis)
        {
            const Extent dist = spec.shape[axis] - 1;
            const offset_type contribution = checked_contribution(layout_.stride(axis), dist);
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

        if (max_offset >= buffer_capacity)
        {
            throw std::invalid_argument{"materialization reaches beyond the end of its buffer"};
        }
    }
}