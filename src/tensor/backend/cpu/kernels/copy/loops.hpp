#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/detail/buffer_access.hpp"
#include "tensor/backend/cpu/iteration/elementwise.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
template <CpuElement T>
void copy_strided(const Shape &shape, const T *source_data, const Layout &source_layout, T *destination_data,
                  const Layout &destination_layout)
{
    assert(source_layout.rank() == shape.rank());
    assert(destination_layout.rank() == shape.rank());

    if (shape.numel() == 0)
    {
        return;
    }

    // Contiguous fast path
    if (source_layout.is_contiguous(shape) && destination_layout.is_contiguous(shape))
    {
        std::copy_n(source_data + static_cast<std::size_t>(source_layout.offset()), shape.numel(),
                    destination_data + static_cast<std::size_t>(destination_layout.offset()));
        return;
    }

    const std::array<Layout, 2> layouts{source_layout, destination_layout};
    const ElementwisePlan iteration(shape, layouts);
    iteration.for_each_run([&](Shape::size_type, std::span<const Layout::offset_type> offsets,
                               std::span<const Layout::stride_type> strides, Shape::size_type run_size) {
        Layout::offset_type source_offset = offsets[0];
        Layout::offset_type destination_offset = offsets[1];
        const Layout::stride_type source_stride = strides[0];
        const Layout::stride_type destination_stride = strides[1];

        // Fast path for single strides
        if (source_stride == 1 && destination_stride == 1)
        {
            assert(source_offset >= 0);
            assert(destination_offset >= 0);
            std::copy_n(source_data + static_cast<std::size_t>(source_offset), run_size,
                        destination_data + static_cast<std::size_t>(destination_offset));
            return;
        }

        for (Shape::size_type i = 0; i < run_size; ++i)
        {
            assert(source_offset >= 0);
            assert(destination_offset >= 0);
            destination_data[static_cast<std::size_t>(destination_offset)] =
                source_data[static_cast<std::size_t>(source_offset)];
            source_offset += source_stride;
            destination_offset += destination_stride;
        }
    });
}

template <CpuElement T>
void copy_to_contiguous(const TensorView &input, MutableTensorView output)
{
    assert(input.dtype() == dtype_of_v<T>);
    assert(output.dtype() == dtype_of_v<T>);
    assert(input.shape().numel() == output.shape().numel());
    assert(output.layout().is_contiguous(output.shape()));

    if (output.shape().numel() == 0)
    {
        return;
    }

    const Layout destination_layout = Layout::contiguous(input.shape(), output.layout().offset());
    copy_strided(input.shape(), data<T>(input), input.layout(), data<T>(output), destination_layout);
}

template <CpuElement T>
void concatenate_copy(std::span<const TensorView> inputs, MutableTensorView output, Shape::size_type axis)
{
    assert(!inputs.empty());
    assert(output.dtype() == dtype_of_v<T>);
    assert(output.layout().is_contiguous(output.shape()));
    assert(axis < output.shape().rank());

    if (output.shape().numel() == 0)
    {
        return;
    }

    T *output_data = data<T>(output);
    Extent axis_start = 0;

    for (const TensorView &input : inputs)
    {
        assert(input.dtype() == dtype_of_v<T>);
        assert(input.shape().rank() == output.shape().rank());

        const Extent input_axis_extent = input.shape()[axis];
        if (input.shape().numel() != 0)
        {
            // Find layout for corresponding output section
            std::vector<Layout::stride_type> output_section_strides(output.layout().strides().begin(),
                                                                    output.layout().strides().end());
            const Layout::offset_type output_section_offset =
                output.layout().offset() + axis_start * output.layout().stride(axis);
            const Layout output_section_layout{std::move(output_section_strides), output_section_offset};

            copy_strided(input.shape(), data<T>(input), input.layout(), output_data, output_section_layout);
        }
        axis_start += input_axis_extent;
    }

    assert(axis_start == output.shape()[axis]);
}

template <CpuElement T>
void basic_index_scatter_copy(const TensorView &input, MutableTensorView output, const Layout &destination_layout)
{
    assert(input.dtype() == dtype_of_v<T>);
    assert(output.dtype() == dtype_of_v<T>);
    assert(output.layout().is_contiguous(output.shape()));
    assert(destination_layout.rank() == input.shape().rank());

    if (output.shape().numel() == 0)
    {
        return;
    }

    T *output_data = data<T>(output);
    assert(output.layout().offset() >= 0);
    const auto output_offset = static_cast<std::size_t>(output.layout().offset());
    std::fill_n(output_data + output_offset, output.shape().numel(), T{0});

    if (input.shape().numel() == 0)
    {
        return;
    }

    copy_strided(input.shape(), data<T>(input), input.layout(), output_data, destination_layout);
}

template <CpuElement T>
void gather_copy(const TensorView &source, const TensorView &indices, MutableTensorView output, Shape::size_type axis)
{
    assert(source.dtype() == dtype_of_v<T>);
    assert(indices.dtype() == DType::Int64);
    assert(output.dtype() == dtype_of_v<T>);
    assert(output.layout().is_contiguous(output.shape()));
    assert(axis < source.shape().rank());
    assert(source.shape().rank() == indices.shape().rank());
    assert(indices.shape() == output.shape());

    if (output.shape().numel() == 0)
    {
        return;
    }

    const T *source_data = data<T>(source);
    const std::int64_t *index_data = data<std::int64_t>(indices);
    T *output_data = data<T>(output);

    // A base layout is constructed with the source strides to
    // account for all non-gather traversal.
    std::vector<Layout::stride_type> source_base_strides(source.layout().strides().begin(),
                                                         source.layout().strides().end());

    const Shape::value_type source_axis_extent = source.shape()[axis];
    const Layout::stride_type source_axis_stride = source.layout().stride(axis);

    // The gather axis index is resolved dynamically.
    source_base_strides[axis] = 0;

    const Layout source_base_layout(std::move(source_base_strides), source.layout().offset());

    // Elementwise iteration over all three layouts supplies
    // the information needed.
    const std::array<Layout, 3> layouts{source_base_layout, indices.layout(), output.layout()};
    const ElementwisePlan iteration(output.shape(), layouts);

    iteration.for_each_run([&](Shape::size_type, std::span<const Layout::offset_type> offsets,
                               std::span<const Layout::stride_type> strides, Shape::size_type run_size) {
        Layout::offset_type source_base_offset = offsets[0];
        Layout::offset_type index_offset = offsets[1];
        Layout::offset_type output_offset = offsets[2];

        for (Shape::size_type i = 0; i < run_size; ++i)
        {
            assert(index_offset >= 0);
            assert(output_offset >= 0);

            const std::int64_t index = index_data[static_cast<std::size_t>(index_offset)];
            if (index < 0 || index >= source_axis_extent)
            {
                throw std::out_of_range{"gather index is out of range"};
            }

            const auto source_gather_offset = source_base_offset + index * source_axis_stride;
            assert(source_gather_offset >= 0);

            output_data[static_cast<std::size_t>(output_offset)] =
                source_data[static_cast<std::size_t>(source_gather_offset)];

            source_base_offset += strides[0];
            index_offset += strides[1];
            output_offset += strides[2];
        }
    });
}

template <CpuElement T>
    requires std::floating_point<T>
void gather_scatter_add(const TensorView &updates, const TensorView &indices, MutableTensorView output,
                        Shape::size_type axis)
{
    assert(updates.dtype() == dtype_of_v<T>);
    assert(indices.dtype() == DType::Int64);
    assert(output.dtype() == dtype_of_v<T>);
    assert(output.layout().is_contiguous(output.shape()));
    assert(axis < output.shape().rank());
    assert(updates.shape().rank() == indices.shape().rank());
    assert(updates.shape().rank() == output.shape().rank());
    assert(updates.shape() == indices.shape());

    const T *updates_data = data<T>(updates);
    const std::int64_t *index_data = data<std::int64_t>(indices);
    T *output_data = data<T>(output);

    // The index and update tensors must be examined even if the
    // target output tensor has no elements, so we cannot simply
    // return on if the output is empty.

    if (output.shape().numel() != 0)
    {
        // Fill the output with zeros to prepare for accumulation
        assert(output.layout().offset() >= 0);
        std::fill_n(output_data + static_cast<std::size_t>(output.layout().offset()), output.shape().numel(), T{0});
    }

    // Empty updates warrants a return because we have already
    // filled the output with zeros.
    if (updates.shape().numel() == 0)
    {
        return;
    }

    // Use the same technique as in the gather kernel, dynamically
    // constructing the gather offset from the target base offset
    // and the indices.
    std::vector<Layout::stride_type> target_base_strides(output.layout().strides().begin(),
                                                         output.layout().strides().end());

    const Shape::value_type target_axis_extent = output.shape()[axis];
    const Layout::stride_type target_axis_stride = output.layout().stride(axis);

    target_base_strides[axis] = 0;

    const Layout target_base_layout{std::move(target_base_strides), output.layout().offset()};

    const std::array<Layout, 3> layouts{updates.layout(), indices.layout(), target_base_layout};
    const ElementwisePlan iteration(updates.shape(), layouts);

    iteration.for_each_run([&](Shape::size_type, std::span<const Layout::offset_type> offsets,
                               std::span<const Layout::stride_type> strides, Shape::size_type run_size) {
        Layout::offset_type updates_offset = offsets[0];
        Layout::offset_type index_offset = offsets[1];
        Layout::offset_type target_base_offset = offsets[2];

        for (Shape::size_type i = 0; i < run_size; ++i)
        {
            assert(updates_offset >= 0);
            assert(index_offset >= 0);

            const std::int64_t index = index_data[static_cast<std::size_t>(index_offset)];
            if (index < 0 || index >= target_axis_extent)
            {
                throw std::out_of_range{"gather scatter add index is out of range"};
            }

            const auto target_offset = target_base_offset + index * target_axis_stride;
            assert(target_offset >= 0);

            // Accumulation is critical because indices may be repeated in gather
            output_data[static_cast<std::size_t>(target_offset)] +=
                updates_data[static_cast<std::size_t>(updates_offset)];

            updates_offset += strides[0];
            index_offset += strides[1];
            target_base_offset += strides[2];
        }
    });
}
} // namespace minitensor::detail::cpu
