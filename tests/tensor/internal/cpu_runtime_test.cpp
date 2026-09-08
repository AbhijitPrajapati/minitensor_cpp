#include "tensor/backend/cpu/cpu_buffer.hpp"
#include "tensor/backend/cpu/cpu_runtime.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "../support/test.hpp"
#include "../support/test_buffer.hpp"

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

        constexpr std::size_t transfer_size = 6;
        const BufferRef transfer_buffer = cpu_runtime.allocate(transfer_size);
        auto *transfer_cpu_buffer = dynamic_cast<CpuBuffer *>(transfer_buffer.get());
        expect(transfer_cpu_buffer != nullptr, "a transfer test requires CPU storage");
        std::fill_n(transfer_cpu_buffer->data(), transfer_size, std::byte{0});

        const std::array<std::byte, 3> source_bytes{
            std::byte{1}, std::byte{2}, std::byte{3}};
        cpu_runtime.copy_from_host(*transfer_buffer, 2, source_bytes);
        const std::array<std::byte, transfer_size> expected_storage{
            std::byte{0}, std::byte{0}, std::byte{1},
            std::byte{2}, std::byte{3}, std::byte{0}};
        expect(std::equal(expected_storage.begin(), expected_storage.end(), transfer_cpu_buffer->data()),
               "copy_from_host writes the requested bytes at the destination offset");

        std::array<std::byte, 2> copied_bytes{};
        cpu_runtime.copy_to_host(copied_bytes, *transfer_buffer, 3);
        expect(copied_bytes == std::array<std::byte, 2>{std::byte{2}, std::byte{3}},
               "copy_to_host reads the requested bytes from the source offset");

        cpu_runtime.copy_from_host(
            *transfer_buffer, transfer_size, std::span<const std::byte>{});
        cpu_runtime.copy_to_host(
            std::span<std::byte>{}, *transfer_buffer, transfer_size);

        expect_throws<std::out_of_range>(
            [&cpu_runtime, &transfer_buffer, &source_bytes]
            {
                cpu_runtime.copy_from_host(
                    *transfer_buffer,
                    transfer_size,
                    std::span{source_bytes}.first<1>());
            },
            "copy_from_host rejects a transfer beyond the destination buffer");
        expect_throws<std::out_of_range>(
            [&cpu_runtime, &transfer_buffer]
            {
                std::array<std::byte, 1> destination{};
                cpu_runtime.copy_to_host(destination, *transfer_buffer, transfer_size);
            },
            "copy_to_host rejects a transfer beyond the source buffer");

        const BufferRef test_buffer = make_test_buffer(transfer_size);
        expect_throws<std::invalid_argument>(
            [&cpu_runtime, &test_buffer]
            {
                cpu_runtime.copy_from_host(*test_buffer, 0, std::span<const std::byte>{});
            },
            "copy_from_host rejects a non-CPU-buffer destination");
        expect_throws<std::invalid_argument>(
            [&cpu_runtime, &test_buffer]
            {
                cpu_runtime.copy_to_host(std::span<std::byte>{}, *test_buffer, 0);
            },
            "copy_to_host rejects a non-CPU-buffer source");

        CpuRuntime other_runtime{Device::cpu(1)};
        const BufferRef other_device_buffer = other_runtime.allocate(transfer_size);
        expect_throws<std::invalid_argument>(
            [&cpu_runtime, &other_device_buffer]
            {
                cpu_runtime.copy_from_host(
                    *other_device_buffer, 0, std::span<const std::byte>{});
            },
            "copy_from_host rejects a buffer belonging to another CPU device");
        expect_throws<std::invalid_argument>(
            [&cpu_runtime, &other_device_buffer]
            {
                cpu_runtime.copy_to_host(
                    std::span<std::byte>{}, *other_device_buffer, 0);
            },
            "copy_to_host rejects a buffer belonging to another CPU device");

        const BufferRef zero_buffer = cpu_runtime.allocate(0);
        expect(zero_buffer != nullptr, "a zero-byte allocation still returns a buffer object");

        auto *cpu_zero_buffer = dynamic_cast<CpuBuffer *>(zero_buffer.get());
        expect(cpu_zero_buffer != nullptr, "a zero-byte cpu allocation returns a cpu buffer");
        expect(zero_buffer->device() == Device::cpu(), "a zero-byte buffer preserves its runtime device");
        expect(zero_buffer->size_bytes() == 0, "a zero-byte buffer reports zero capacity");
        expect(cpu_zero_buffer->data() == nullptr, "a zero-byte cpu buffer has no storage pointer");
    }
}
