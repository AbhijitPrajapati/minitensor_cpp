#include "tensor/backend/cpu/cpu_buffer.hpp"
#include "tensor/backend/cpu/cpu_runtime.hpp"

#include <cstddef>
#include <cstdint>

#include "../support/test.hpp"

namespace minitensor::test
{
    void run_cpu_runtime_test()
    {
        using detail::BufferRef;
        using detail::cpu::CpuBuffer;
        using detail::cpu::CpuRuntime;

        CpuRuntime cpu_runtime;
        expect(cpu_runtime.device() == Device::cpu(), "default cpu runtime reports cpu device zero");

        constexpr std::size_t buffer_size = 2 * sizeof(float);
        const BufferRef buffer = cpu_runtime.allocate(buffer_size);
        expect(buffer != nullptr, "a nonzero allocation returns a buffer");

        auto *cpu_buffer = dynamic_cast<CpuBuffer *>(buffer.get());
        expect(cpu_buffer != nullptr, "a cpu runtime allocates cpu buffers");

        expect(buffer->device() == cpu_runtime.device(), "an allocated buffer preserves its runtime device");
        expect(buffer->size_bytes() == buffer_size, "a cpu buffer preserves an arbitrary byte size");
        expect(cpu_buffer->data() != nullptr, "a nonzero cpu buffer owns storage");

        const auto address = reinterpret_cast<std::uintptr_t>(cpu_buffer->data());
        expect(address % CpuBuffer::alignment == 0, "cpu buffer is correctly aligned");

        auto *values = reinterpret_cast<float *>(cpu_buffer->data());
        values[0] = 1.0F;
        values[1] = 2.0F;
        expect(values[0] == 1.0F && values[1] == 2.0F, "cpu buffer can be read and written");

        const BufferRef zero_buffer = cpu_runtime.allocate(0);
        expect(zero_buffer != nullptr, "a zero-byte allocation still returns a buffer object");

        auto *cpu_zero_buffer = dynamic_cast<CpuBuffer *>(zero_buffer.get());
        expect(cpu_zero_buffer != nullptr, "a zero-byte cpu allocation returns a cpu buffer");
        expect(zero_buffer->device() == Device::cpu(), "a zero-byte buffer preserves its runtime device");
        expect(zero_buffer->size_bytes() == 0, "a zero-byte buffer reports zero capacity");
        expect(cpu_zero_buffer->data() == nullptr, "a zero-byte cpu buffer has no storage pointer");
    }
}
