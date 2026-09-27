#include <minitensor/types.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include <gtest/gtest.h>

#include "tensor/backend/cpu/cpu_buffer.hpp"
#include "tensor/backend/cpu/cpu_runtime.hpp"
#include "tensor/support/test_buffer.hpp"

namespace minitensor::test
{
	TEST(CpuRuntimeTest, AllocatesAlignedBuffersForItsDevice)
	{
		detail::cpu::CpuRuntime runtime{ Device::cpu(2) };
		const detail::BufferRef buffer = runtime.allocate(2 * sizeof(float));
		ASSERT_NE(buffer, nullptr);
		const auto* cpu_buffer = dynamic_cast<const detail::cpu::CpuBuffer*>(buffer.get());
		ASSERT_NE(cpu_buffer, nullptr);

		EXPECT_EQ(runtime.device(), Device::cpu(2));
		EXPECT_EQ(buffer->device(), runtime.device());
		EXPECT_EQ(buffer->size_bytes(), 2 * sizeof(float));
		ASSERT_NE(cpu_buffer->data(), nullptr);
		const auto address = reinterpret_cast<std::uintptr_t>(cpu_buffer->data());
		EXPECT_EQ(address % detail::cpu::CpuBuffer::alignment, 0);
	}

	TEST(CpuRuntimeTest, CopiesBytesAtExplicitOffsets)
	{
		detail::cpu::CpuRuntime runtime;
		const detail::BufferRef buffer = runtime.allocate(6);
		auto* cpu_buffer = dynamic_cast<detail::cpu::CpuBuffer*>(buffer.get());
		ASSERT_NE(cpu_buffer, nullptr);
		std::fill_n(cpu_buffer->data(), 6, std::byte{ 0 });

		const std::array<std::byte, 3> source{
			std::byte{ 1 }, std::byte{ 2 }, std::byte{ 3 } };
		runtime.copy_from_host(*buffer, 2, source);
		const std::array<std::byte, 6> expected{
			std::byte{ 0 }, std::byte{ 0 }, std::byte{ 1 },
			std::byte{ 2 }, std::byte{ 3 }, std::byte{ 0 } };
		EXPECT_TRUE(std::equal(expected.begin(), expected.end(), cpu_buffer->data()));

		std::array<std::byte, 2> destination{};
		runtime.copy_to_host(destination, *buffer, 3);
		EXPECT_EQ(destination, (std::array<std::byte, 2>{ std::byte{ 2 }, std::byte{ 3 } }));
	}

	TEST(CpuRuntimeTest, AcceptsEmptyCopiesAtTheBufferBoundary)
	{
		detail::cpu::CpuRuntime runtime;
		const detail::BufferRef buffer = runtime.allocate(4);
		EXPECT_NO_THROW(runtime.copy_from_host(
			*buffer, 4, std::span<const std::byte>{}));
		EXPECT_NO_THROW(runtime.copy_to_host(
			std::span<std::byte>{}, *buffer, 4));
	}

	TEST(CpuRuntimeTest, ValidatesTransferBoundsAndBufferKinds)
	{
		detail::cpu::CpuRuntime runtime;
		const detail::BufferRef buffer = runtime.allocate(4);
		const std::array<std::byte, 1> byte{ std::byte{ 1 } };
		EXPECT_THROW(runtime.copy_from_host(*buffer, 4, byte), std::out_of_range);

		std::array<std::byte, 1> destination{};
		EXPECT_THROW(runtime.copy_to_host(destination, *buffer, 4), std::out_of_range);

		const detail::BufferRef foreign_kind = make_test_buffer(4);
		EXPECT_THROW(
			runtime.copy_from_host(*foreign_kind, 0, std::span<const std::byte>{}),
			std::invalid_argument);
		EXPECT_THROW(
			runtime.copy_to_host(std::span<std::byte>{}, *foreign_kind, 0),
			std::invalid_argument);
	}

	TEST(CpuRuntimeTest, RejectsBuffersFromAnotherCpuDevice)
	{
		detail::cpu::CpuRuntime runtime;
		detail::cpu::CpuRuntime other_runtime{ Device::cpu(1) };
		const detail::BufferRef buffer = other_runtime.allocate(4);
		EXPECT_THROW(
			runtime.copy_from_host(*buffer, 0, std::span<const std::byte>{}),
			std::invalid_argument);
		EXPECT_THROW(
			runtime.copy_to_host(std::span<std::byte>{}, *buffer, 0),
			std::invalid_argument);
	}

	TEST(CpuRuntimeTest, RepresentsZeroByteAllocations)
	{
		detail::cpu::CpuRuntime runtime;
		const detail::BufferRef buffer = runtime.allocate(0);
		ASSERT_NE(buffer, nullptr);
		const auto* cpu_buffer = dynamic_cast<const detail::cpu::CpuBuffer*>(buffer.get());
		ASSERT_NE(cpu_buffer, nullptr);
		EXPECT_EQ(buffer->size_bytes(), 0);
		EXPECT_EQ(cpu_buffer->data(), nullptr);
	}
}
