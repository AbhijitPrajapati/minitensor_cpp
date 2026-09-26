#include "evaluator.hpp"

#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "runtime_registry.hpp"
#include "tensor/backend/device_runtime.hpp"
#include "tensor/core/dense_size.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/node.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/storage/buffer.hpp"
#include "tensor/storage/layout.hpp"
#include "tensor/storage/materialization.hpp"

namespace minitensor::detail
{
	namespace
	{
		struct PlannedKernel
		{
			DeviceRuntime* runtime;
			KernelFn function;
		};
		struct PlannedStep final
		{
			ValueRef output;
			NodeRef node;
			std::optional<PlannedKernel> kernel;
		};

		using EvaluationPlan = std::vector<PlannedStep>;

		enum class VisitState
		{
			Visiting,
			Finished
		};

		class EvaluationPlanner final
		{
		public:
			EvaluationPlanner(const RuntimeRegistry& runtimes, const KernelRegistry& kernels) noexcept : runtimes_{ runtimes }, kernels_{ kernels } {}
			EvaluationPlan build(std::span<const ValueRef> roots)
			{
				for (const ValueRef& value : roots)
				{
					visit(value);
				}
				return std::move(plan_);
			}

		private:
			void visit(const ValueRef& value)
			{
				if (!value)
				{
					throw std::invalid_argument{ "cannot evaluate an empty value" };
				}

				// Cached value
				if (value->materialization() != nullptr)
				{
					return;
				}

				const auto existing = states_.find(value.get());
				if (existing != states_.end())
				{
					// Cycle in the graph if value has already been visited
					if (existing->second == VisitState::Visiting)
					{
						throw std::logic_error{ "cycle in computation graph" };
					}
					return;
				}

				states_.emplace(value.get(), VisitState::Visiting);

				const NodeRef& node = value->producer_ref();
				// No producer and no materialization
				if (!node)
				{
					throw std::runtime_error{ "cannot evaluate unmaterialized leaf" };
				}

				// Recursively visit input values
				for (const ValueRef& input : node->inputs())
				{
					visit(input);
				}

				// Add execution details to the plan
				const Primitive& primitive = node->primitive();

				std::optional<PlannedKernel> kernel;

				// Detect kernel requirement
				if (primitive.requires_kernel_support())
				{
					const TensorSpec& output_spec = value->spec();
					DeviceRuntime& runtime = runtimes_.get(output_spec.device);
					const KernelKey key{ typeid(node->primitive()), output_spec.device.type() };
					kernel = PlannedKernel{ &runtime, kernels_.get(key) };
				}

				plan_.push_back(PlannedStep{ value, node, std::move(kernel) });
				states_.at(value.get()) = VisitState::Finished;
			}

			const RuntimeRegistry& runtimes_;
			const KernelRegistry& kernels_;
			std::unordered_map<const Value*, VisitState> states_;
			EvaluationPlan plan_;
		};

		void execute_step(const PlannedStep& step)
		{
			Value& output = *step.output;
			const Node& node = *step.node;
			const Primitive& primitive = node.primitive();
			const auto& inputs = node.inputs();

			// Construct input view
			std::vector<TensorView> input_views;
			input_views.reserve(inputs.size());

			for (const ValueRef& input : inputs)
			{
				const Materialization* materialization = input->materialization();
				if (materialization == nullptr)
				{
					throw std::logic_error{ "planned operation has unmaterialized input" };
				}

				// Storage-sharing operations handled once here
				if (inputs.size() == 1)
				{
					// Only check for storage sharing from single-input operations
					// All storage-sharing operations must have a singular input
					auto shared_layout = primitive.try_derive_shared_layout(input->spec(), materialization->layout(), output.spec());
					if (shared_layout)
					{
						Materialization output_materialization = Materialization(materialization->buffer_ref(), std::move(shared_layout.value()));
						output.materialize(std::move(output_materialization));
						return;
					}
				}

				input_views.emplace_back(input->spec(), *materialization);
			}

			// Storage-sharing operations should have already been handled by now
			// All non-storage-sharing operations should provide kernels
			if (!step.kernel)
			{
				throw std::logic_error{ "primitive must either share storage or plan a kernel" };
			}

			const PlannedKernel& planned_kernel = step.kernel.value();
			const TensorSpec& output_spec = output.spec();
			BufferRef output_buffer = planned_kernel.runtime->allocate(dense_size_bytes(output_spec));
			Materialization output_materialization(std::move(output_buffer), Layout::contiguous(output_spec.shape));

			// Perform operation
			{
				MutableTensorView output_view(output_spec, output_materialization);
				planned_kernel.function(*planned_kernel.runtime, primitive, input_views, output_view);
			}

			// Only materialize once operation succeeds
			output.materialize(std::move(output_materialization));
		}

	}

	Evaluator::Evaluator(RuntimeRegistry& runtimes, const KernelRegistry& kernels) noexcept : runtimes_{ runtimes }, kernels_{ kernels } {}

	void Evaluator::evaluate(std::span<const ValueRef> roots)
	{
		EvaluationPlanner planner(runtimes_, kernels_);
		EvaluationPlan plan = planner.build(roots);
		for (const PlannedStep& step : plan)
		{
			execute_step(step);
		}
	}
}