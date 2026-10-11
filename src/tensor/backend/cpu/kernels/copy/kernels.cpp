#include "tensor/backend/cpu/kernels/registrations.hpp"

#include <cassert>
#include <span>
#include <type_traits>

#include <minitensor/types.hpp>

#include "loops.hpp"
#include "tensor/backend/cpu/detail/dtype_dispatch.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/primitives/manipulation/basic_index.hpp"
#include "tensor/primitives/manipulation/concatenate.hpp"
#include "tensor/primitives/manipulation/contiguous.hpp"
#include "tensor/primitives/manipulation/reshape.hpp"

namespace minitensor::detail::cpu
{
	namespace
	{
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

		void run_concatenate(
			DeviceRuntime&,
			const Primitive& primitive,
			std::span<const TensorView> inputs,
			MutableTensorView output)
		{
			assert(!inputs.empty());
			const auto& concatenate_primitive = dynamic_cast<const ConcatenatePrimitive&>(primitive);

			dispatch_dtype(
				output.dtype(),
				[&]<typename T>(std::type_identity<T>)
			{
				concatenate_copy<T>(inputs, output, concatenate_primitive.axis());
			});
		}

		void run_basic_index_scatter(
			DeviceRuntime&,
			const Primitive& primitive,
			std::span<const TensorView> inputs,
			MutableTensorView output)
		{
			assert(inputs.size() == 1);
			const auto& basic_index_scatter = dynamic_cast<const BasicIndexScatterPrimitive&>(primitive);
			const BasicIndexParameters& parameters = basic_index_scatter.parameters();
			assert(inputs.front().shape() == parameters.output_shape());
			assert(output.shape() == parameters.input_shape());

			dispatch_dtype(
				output.dtype(), [&]<typename T>(std::type_identity<T>)
			{
				basic_index_scatter_copy<T>(
					inputs.front(),
					output,
					parameters.derive_layout(output.layout()));
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
		registry.register_kernel(
			KernelKey{ typeid(ConcatenatePrimitive), DeviceType::Cpu },
			run_concatenate);
		registry.register_kernel(
			KernelKey{ typeid(BasicIndexScatterPrimitive), DeviceType::Cpu },
			run_basic_index_scatter);
	}
}
