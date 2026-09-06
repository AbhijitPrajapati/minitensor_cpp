#include <minitensor/types.hpp>

#include <stdexcept>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

#include "../support/test.hpp"
#include "../support/test_buffer.hpp"

namespace minitensor::test
{
    void run_tensor_view_test()
    {
        using detail::BufferRef;
        using detail::Layout;
        using detail::Materialization;
        using detail::MutableTensorView;
        using detail::TensorSpec;
        using detail::TensorView;

        const TensorSpec spec{Shape{2, 2}, DType::Float32, Device::cpu(2)};
        const Layout layout{{3, 1}, 1};
        const BufferRef buffer = make_test_buffer(6 * sizeof(float), Device::cpu(2));
        const Materialization materialization{buffer, layout};

        const TensorView view{spec, materialization};
        expect(view.shape() == spec.shape, "a tensor view exposes its shape");
        expect(view.dtype() == spec.dtype, "a tensor view exposes its dtype");
        expect(view.device() == spec.device, "a tensor view exposes its device");
        expect(view.layout() == layout, "a tensor view exposes its materialization layout");
        expect(&view.buffer() == buffer.get(), "a tensor view exposes its buffer as read-only storage");

        const MutableTensorView mutable_view{spec, materialization};
        expect(mutable_view.shape() == spec.shape, "a mutable tensor view exposes its shape");
        expect(mutable_view.dtype() == spec.dtype, "a mutable tensor view exposes its dtype");
        expect(mutable_view.device() == spec.device, "a mutable tensor view exposes its device");
        expect(mutable_view.layout() == layout, "a mutable tensor view exposes its materialization layout");
        expect(&mutable_view.buffer() == buffer.get(), "a mutable tensor view exposes its writable buffer");

        expect_throws<std::invalid_argument>(
            [&spec]
            {
                const Materialization wrong_rank{make_test_buffer(6 * sizeof(float), spec.device), Layout({1})};
                const TensorView invalid{spec, wrong_rank};
                (void)invalid;
            },
            "tensor view construction validates the materialization rank");

        expect_throws<std::invalid_argument>(
            []
            {
                const TensorSpec cpu_zero_spec{Shape{2}, DType::Float32, Device::cpu(0)};
                const Materialization cpu_one_materialization{
                    make_test_buffer(2 * sizeof(float), Device::cpu(1)),
                    Layout::contiguous(cpu_zero_spec.shape)};
                const MutableTensorView invalid{cpu_zero_spec, cpu_one_materialization};
                (void)invalid;
            },
            "mutable tensor view construction validates the materialization device");
    }
}
