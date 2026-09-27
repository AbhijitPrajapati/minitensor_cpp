#include <minitensor/types.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <typeinfo>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/execution/evaluator.hpp"
#include "tensor/execution/runtime_registry.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/graph/node.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/storage/buffer.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"
#include "tensor/support/test_buffer.hpp"
#include "tensor/support/test_primitive.hpp"
#include "tensor/support/test_runtime.hpp"

namespace minitensor::test
{
	namespace
	{
		detail::ValueRef make_materialized_leaf(const detail::TensorSpec& spec)
		{
			const std::size_t size_bytes = spec.shape.numel() * dtype_size(spec.dtype);
			auto value = std::make_shared<detail::Value>(spec);
			value->materialize(detail::Materialization{
				make_test_buffer(size_bytes, spec.device),
				detail::Layout::contiguous(spec.shape) });
			return value;
		}

		detail::ValueRef apply_identity(const detail::ValueRef& input)
		{
			const std::array<detail::ValueRef, 1> inputs{ input };
			return detail::apply_operation(
				std::make_unique<IdentitySpecPrimitive>(), inputs);
		}
	}

	class EvaluatorTest : public testing::Test
	{
	protected:
		struct KernelCall final
		{
			detail::DeviceRuntime* runtime;
			const detail::Primitive* primitive;
			const detail::Buffer* input_buffer;
			detail::Buffer* output_buffer;
			Shape output_shape;
			detail::Layout output_layout;
		};

		void SetUp() override
		{
			auto runtime = std::make_unique<TestRuntime>();
			runtime_ = runtime.get();
			runtimes_.register_runtime(std::move(runtime));
			kernels_.register_kernel(
				identity_key_,
				[this](
					detail::DeviceRuntime& runtime,
					const detail::Primitive& primitive,
					std::span<const detail::TensorView> inputs,
					detail::MutableTensorView output)
				{
					ASSERT_EQ(inputs.size(), 1);
					calls_.push_back(KernelCall{
						&runtime,
						&primitive,
						&inputs.front().buffer(),
						&output.buffer(),
						output.shape(),
						output.layout() });
				});
		}

		const detail::TensorSpec spec_{ Shape{ 2, 3 }, DType::Float32, Device::cpu() };
		const detail::KernelKey identity_key_{
			typeid(IdentitySpecPrimitive), DeviceType::Cpu };
		detail::RuntimeRegistry runtimes_;
		detail::KernelRegistry kernels_;
		TestRuntime* runtime_{ nullptr };
		std::vector<KernelCall> calls_;
	};

	TEST_F(EvaluatorTest, PerformsNoWorkForEmptyOrCachedRoots)
	{
		detail::Evaluator evaluator{ runtimes_, kernels_ };
		evaluator.evaluate(std::span<const detail::ValueRef>{});
		const detail::ValueRef leaf = make_materialized_leaf(spec_);
		evaluator.evaluate(std::array<detail::ValueRef, 1>{ leaf });

		EXPECT_TRUE(calls_.empty());
		EXPECT_TRUE(runtime_->allocation_sizes().empty());
	}

	TEST_F(EvaluatorTest, ExecutesSharedGraphsOnceInTopologicalOrder)
	{
		const detail::ValueRef leaf = make_materialized_leaf(spec_);
		const detail::ValueRef intermediate = apply_identity(leaf);
		const detail::ValueRef output = apply_identity(intermediate);
		const std::array<detail::ValueRef, 3> roots{ intermediate, output, output };

		detail::Evaluator evaluator{ runtimes_, kernels_ };
		evaluator.evaluate(roots);

		ASSERT_EQ(calls_.size(), 2);
		ASSERT_EQ(runtime_->allocation_sizes().size(), 2);
		EXPECT_EQ(runtime_->allocation_sizes()[0], 6 * sizeof(float));
		EXPECT_EQ(runtime_->allocation_sizes()[1], 6 * sizeof(float));
		ASSERT_NE(intermediate->materialization(), nullptr);
		ASSERT_NE(output->materialization(), nullptr);
		EXPECT_EQ(calls_[0].runtime, runtime_);
		EXPECT_EQ(calls_[1].runtime, runtime_);
		EXPECT_EQ(calls_[0].primitive, &intermediate->producer()->primitive());
		EXPECT_EQ(calls_[1].primitive, &output->producer()->primitive());
		EXPECT_EQ(calls_[0].input_buffer, leaf->materialization()->buffer_ref().get());
		EXPECT_EQ(calls_[1].input_buffer, intermediate->materialization()->buffer_ref().get());
		EXPECT_EQ(calls_[0].output_buffer, intermediate->materialization()->buffer_ref().get());
		EXPECT_EQ(calls_[1].output_buffer, output->materialization()->buffer_ref().get());
		EXPECT_EQ(calls_[0].output_shape, spec_.shape);
		EXPECT_EQ(calls_[0].output_layout, detail::Layout::contiguous(spec_.shape));

		evaluator.evaluate(std::array<detail::ValueRef, 1>{ output });
		EXPECT_EQ(calls_.size(), 2);
	}

	TEST(EvaluatorStorageSharingTest, RequiresNeitherRuntimeNorKernel)
	{
		const detail::TensorSpec spec{ Shape{ 2, 3 }, DType::Float32, Device::cpu() };
		const detail::ValueRef input = std::make_shared<detail::Value>(spec);
		input->materialize(detail::Materialization{
			make_test_buffer(6 * sizeof(float)), detail::Layout{ { 1, 2 } } });
		const std::array<detail::ValueRef, 1> inputs{ input };
		const detail::ValueRef output = detail::apply_operation(
			std::make_unique<IdentityStorageSharingPrimitive>(), inputs);

		detail::RuntimeRegistry runtimes;
		detail::KernelRegistry kernels;
		detail::Evaluator evaluator{ runtimes, kernels };
		evaluator.evaluate(std::array<detail::ValueRef, 1>{ output });

		ASSERT_NE(output->materialization(), nullptr);
		EXPECT_EQ(
			output->materialization()->buffer_ref(),
			input->materialization()->buffer_ref());
		EXPECT_EQ(output->materialization()->layout(), (detail::Layout{ { 1, 2 } }));
	}

	TEST_F(EvaluatorTest, RejectsNullAndUnmaterializedLeaves)
	{
		detail::Evaluator evaluator{ runtimes_, kernels_ };
		const std::array<detail::ValueRef, 1> null_roots{ detail::ValueRef{} };
		EXPECT_THROW(evaluator.evaluate(null_roots), std::invalid_argument);

		const auto leaf = std::make_shared<detail::Value>(spec_);
		EXPECT_THROW(
			evaluator.evaluate(std::array<detail::ValueRef, 1>{ leaf }),
			std::runtime_error);
	}

	TEST(EvaluatorPlanningTest, MissingRuntimeLeavesOutputUntouched)
	{
		const detail::TensorSpec spec{ Shape{ 2 }, DType::Float32, Device::cpu() };
		const detail::ValueRef output = apply_identity(make_materialized_leaf(spec));
		detail::RuntimeRegistry runtimes;
		detail::KernelRegistry kernels;
		kernels.register_kernel(
			{ typeid(IdentitySpecPrimitive), DeviceType::Cpu },
			[](detail::DeviceRuntime&, const detail::Primitive&,
				std::span<const detail::TensorView>, detail::MutableTensorView) {});

		detail::Evaluator evaluator{ runtimes, kernels };
		EXPECT_THROW(
			evaluator.evaluate(std::array<detail::ValueRef, 1>{ output }),
			std::runtime_error);
		EXPECT_EQ(output->materialization(), nullptr);
	}

	TEST(EvaluatorPlanningTest, MissingKernelFailsBeforeAnyExecution)
	{
		const detail::TensorSpec spec{ Shape{ 2 }, DType::Float32, Device::cpu() };
		const detail::ValueRef leaf = make_materialized_leaf(spec);
		const detail::ValueRef valid = apply_identity(leaf);
		const std::array<detail::ValueRef, 1> inputs{ leaf };
		bool destroyed = false;
		const detail::ValueRef unavailable = detail::apply_operation(
			std::make_unique<DestructionTrackedPrimitive>(destroyed), inputs);

		detail::RuntimeRegistry runtimes;
		auto runtime = std::make_unique<TestRuntime>();
		TestRuntime* runtime_address = runtime.get();
		runtimes.register_runtime(std::move(runtime));
		detail::KernelRegistry kernels;
		std::size_t calls = 0;
		kernels.register_kernel(
			{ typeid(IdentitySpecPrimitive), DeviceType::Cpu },
			[&calls](detail::DeviceRuntime&, const detail::Primitive&,
				std::span<const detail::TensorView>, detail::MutableTensorView)
			{
				++calls;
			});

		detail::Evaluator evaluator{ runtimes, kernels };
		const std::array<detail::ValueRef, 2> roots{ valid, unavailable };
		EXPECT_THROW(evaluator.evaluate(roots), std::runtime_error);
		EXPECT_EQ(calls, 0);
		EXPECT_TRUE(runtime_address->allocation_sizes().empty());
		EXPECT_EQ(valid->materialization(), nullptr);
		EXPECT_EQ(unavailable->materialization(), nullptr);
	}

	TEST(EvaluatorExecutionTest, KernelFailureDoesNotInstallPartialStorage)
	{
		const detail::TensorSpec spec{ Shape{ 2 }, DType::Float32, Device::cpu() };
		const detail::ValueRef output = apply_identity(make_materialized_leaf(spec));
		detail::RuntimeRegistry runtimes;
		auto runtime = std::make_unique<TestRuntime>();
		TestRuntime* runtime_address = runtime.get();
		runtimes.register_runtime(std::move(runtime));
		detail::KernelRegistry kernels;
		kernels.register_kernel(
			{ typeid(IdentitySpecPrimitive), DeviceType::Cpu },
			[](detail::DeviceRuntime&, const detail::Primitive&,
				std::span<const detail::TensorView>, detail::MutableTensorView)
			{
				throw std::runtime_error{ "test failure" };
			});

		detail::Evaluator evaluator{ runtimes, kernels };
		EXPECT_THROW(
			evaluator.evaluate(std::array<detail::ValueRef, 1>{ output }),
			std::runtime_error);
		EXPECT_EQ(runtime_address->allocation_sizes().size(), 1);
		EXPECT_EQ(output->materialization(), nullptr);
	}
}
