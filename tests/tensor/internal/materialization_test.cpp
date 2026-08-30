#include <minitensor/types.hpp>

#include <stdexcept>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/ids.hpp"
#include "tensor/graph/origin.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

#include "../support/test.hpp"
#include "../support/test_buffer.hpp"

namespace minitensor::test
{
    void run_materialization_test()
    {
        using detail::BufferRef;
        using detail::Layout;
        using detail::LeafOrigin;
        using detail::Materialization;
        using detail::TensorSpec;
        using detail::Value;
        using detail::ValueId;

        const BufferRef buffer = make_test_buffer(6 * sizeof(float), Device::cpu(2));
        expect(buffer->device() == Device::cpu(2), "a buffer reports its device");
        expect(buffer->size_bytes() == 6 * sizeof(float), "a buffer reports its byte capacity");

        expect_throws<std::invalid_argument>(
            []
            {
                const Materialization invalid{BufferRef{}, Layout{}};
                (void)invalid;
            },
            "materialization construction rejects a null buffer");

        const Layout contiguous_layout = Layout::contiguous(Shape{2, 3});
        const Materialization contiguous{buffer, contiguous_layout};
        expect(contiguous.buffer_ref().get() == buffer.get(), "materialization retains the supplied buffer");
        expect(contiguous.layout() == contiguous_layout, "materialization retains the supplied layout");

        const TensorSpec contiguous_spec{Shape{2, 3}, DType::Float32, Device::cpu(2)};
        contiguous.validate(contiguous_spec);

        const Materialization reversed{make_test_buffer(6 * sizeof(float)), Layout({-3, 1}, 3)};
        reversed.validate(TensorSpec{Shape{2, 3}, DType::Float32, Device::cpu()});

        const Materialization empty{make_test_buffer(0), Layout({9, 1}, 50)};
        empty.validate(TensorSpec{Shape{0, 3}, DType::Float32, Device::cpu()});

        expect_throws<std::invalid_argument>(
            []
            {
                const Materialization wrong_device{make_test_buffer(2 * sizeof(float), Device::cpu(1)),
                                                   Layout::contiguous(Shape{2})};
                wrong_device.validate(TensorSpec{Shape{2}, DType::Float32, Device::cpu(0)});
            },
            "materialization validation rejects a buffer on the wrong device");
        expect_throws<std::invalid_argument>(
            []
            {
                const Materialization wrong_rank{make_test_buffer(6 * sizeof(float)), Layout({1})};
                wrong_rank.validate(TensorSpec{Shape{2, 3}, DType::Float32, Device::cpu()});
            },
            "materialization validation rejects a layout with the wrong rank");
        expect_throws<std::invalid_argument>(
            []
            {
                const Materialization too_small{make_test_buffer(5 * sizeof(float)),
                                                Layout::contiguous(Shape{2, 3})};
                too_small.validate(TensorSpec{Shape{2, 3}, DType::Float32, Device::cpu()});
            },
            "materialization validation rejects access beyond the buffer");
        expect_throws<std::invalid_argument>(
            []
            {
                const Materialization before_start{make_test_buffer(2 * sizeof(float)), Layout({-1})};
                before_start.validate(TensorSpec{Shape{2}, DType::Float32, Device::cpu()});
            },
            "materialization validation rejects access before the buffer");

        Value value{ValueId{300}, contiguous_spec, LeafOrigin{}};
        expect(value.materialization() == nullptr, "a new value is not materialized");
        value.materialize(Materialization{buffer, contiguous_layout});
        expect(value.materialization() != nullptr, "materialize installs storage on a value");
        expect(value.materialization()->buffer_ref().get() == buffer.get(),
               "a value exposes its installed materialization");
        expect_throws<std::logic_error>(
            [&value, &buffer, &contiguous_layout]
            {
                value.materialize(Materialization{buffer, contiguous_layout});
            },
            "a value cannot be materialized twice");

        Value rejected{ValueId{301}, TensorSpec{Shape{2}, DType::Float32, Device::cpu()}, LeafOrigin{}};
        expect_throws<std::invalid_argument>(
            [&rejected]
            {
                rejected.materialize(
                    Materialization{make_test_buffer(sizeof(float)), Layout::contiguous(Shape{2})});
            },
            "a value rejects an invalid materialization");
        expect(rejected.materialization() == nullptr,
               "failed materialization validation leaves the value unmaterialized");
    }
}
