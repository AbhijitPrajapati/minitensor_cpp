#include <minitensor/types.hpp>

#include <array>
#include <typeindex>

#include <gtest/gtest.h>

#include "tensor/backend/cpu/register_kernels.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/primitives/creation/full.hpp"
#include "tensor/primitives/creation/normal.hpp"
#include "tensor/primitives/creation/uniform.hpp"
#include "tensor/primitives/elementwise/add.hpp"
#include "tensor/primitives/elementwise/divide.hpp"
#include "tensor/primitives/elementwise/exponential.hpp"
#include "tensor/primitives/elementwise/hyperbolic_tangent.hpp"
#include "tensor/primitives/elementwise/logarithm.hpp"
#include "tensor/primitives/elementwise/multiply.hpp"
#include "tensor/primitives/elementwise/negate.hpp"
#include "tensor/primitives/elementwise/square_root.hpp"
#include "tensor/primitives/elementwise/subtract.hpp"
#include "tensor/primitives/linalg/matmul.hpp"
#include "tensor/primitives/manipulation/concatenate.hpp"
#include "tensor/primitives/manipulation/contiguous.hpp"
#include "tensor/primitives/manipulation/reshape.hpp"
#include "tensor/primitives/manipulation/slice.hpp"
#include "tensor/primitives/reduction/sum.hpp"

namespace minitensor::test
{
	TEST(CpuKernelRegistrationTest, RegistersEveryKernelBackedPrimitive)
	{
		detail::KernelRegistry registry;
		detail::cpu::register_kernels(registry);
		const std::array<std::type_index, 17> primitive_types{
			typeid(detail::FullPrimitive),
			typeid(detail::UniformPrimitive),
			typeid(detail::NormalPrimitive),
			typeid(detail::NegatePrimitive),
			typeid(detail::ExponentialPrimitive),
			typeid(detail::LogarithmPrimitive),
			typeid(detail::SquareRootPrimitive),
			typeid(detail::HyperbolicTangentPrimitive),
			typeid(detail::AddPrimitive),
			typeid(detail::SubtractPrimitive),
			typeid(detail::MultiplyPrimitive),
			typeid(detail::DividePrimitive),
			typeid(detail::ContiguousPrimitive),
			typeid(detail::ReshapePrimitive),
			typeid(detail::ConcatenatePrimitive),
			typeid(detail::SliceScatterPrimitive),
			typeid(detail::SumPrimitive)
		};

		for (const std::type_index primitive_type : primitive_types)
		{
			EXPECT_TRUE(registry.contains(
				detail::KernelKey{ primitive_type, DeviceType::Cpu }))
				<< primitive_type.name();
		}

		EXPECT_TRUE(registry.contains(detail::KernelKey{
			typeid(detail::MatmulPrimitive), DeviceType::Cpu }));
	}
}
