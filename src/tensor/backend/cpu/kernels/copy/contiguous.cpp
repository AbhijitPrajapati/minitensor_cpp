#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <span>
#include <type_traits>

#include <minitensor/types.hpp>

#include "tensor/backend/cpu/detail/buffer_access.hpp"
#include "tensor/backend/cpu/detail/dtype_dispatch.hpp"
#include "tensor/backend/cpu/iteration/elementwise.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/primitives/manipulation/contiguous.hpp"
#include "tensor/primitives/manipulation/reshape.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::detail::cpu
{
	namespace
	{
		template <CpuElement T>
		void copy_to_contiguous(const TensorView& input, MutableTensorView output)
		{
			assert(input.dtype() == ElementDType<T>::value);
			assert(output.dtype() == ElementDType<T>::value);
			assert(input.shape().numel() == output.shape().numel());
			assert(output.layout().is_contiguous(output.shape()));

			if (output.shape().numel() == 0)
			{
				return;
			}

			const T* input_data = data<T>(input);
			T* output_data = data<T>(output);
			const auto input_offset = static_cast<std::size_t>(input.layout().offset());
			const auto output_offset = static_cast<std::size_t>(output.layout().offset());
			const Shape::size_type numel = output.shape().numel();

			// Contiguous fast path
			if (input.layout().is_contiguous(input.shape()))
			{
				std::copy_n(input_data + input_offset, numel, output_data + output_offset);
				return;
			}

			// Regular path
			const std::array<Layout, 1> layouts{ input.layout() };
			const ElementwisePlan iteration(input.shape(), layouts);
			iteration.for_each_run(
				[&](Shape::size_type linear,
				std::span<const Layout::offset_type> offsets,
				std::span<const Layout::stride_type> strides,
				Shape::size_type run_size)
				{
					Layout::offset_type input_offset = offsets[0];
					const Layout::stride_type input_stride = strides[0];
					T* run_output = output_data + output_offset + linear;
					for (Shape::size_type i = 0; i < run_size; ++i)
					{
						assert(input_offset >= 0);
						run_output[i] = input_data[static_cast<std::size_t>(input_offset)];
						input_offset += input_stride;
					}
				});
		}

		void run_contiguous(
			DeviceRuntime&,
			const Primitive& primitive,
			std::span<const TensorView> inputs,
			MutableTensorView output)
		{
			assert(inputs.size() == 1);
			assert(
				dynamic_cast<const ContiguousPrimitive*>(&primitive) != nullptr ||
				dynamic_cast<const ReshapePrimitive*>(&primitive) != nullptr);

			dispatch_dtype(
				output.dtype(),
				[&]<typename T>(std::type_identity<T>)
			{
				copy_to_contiguous<T>(inputs.front(), output);
			});
		}
	}

	void register_copy_kernels(KernelRegistry& registry)
	{
		registry.register_kernel(
			KernelKey{ typeid(ContiguousPrimitive), DeviceType::Cpu },
			run_contiguous);
		registry.register_kernel(
			KernelKey{ typeid(ReshapePrimitive), DeviceType::Cpu },
			run_contiguous);
	}
}
