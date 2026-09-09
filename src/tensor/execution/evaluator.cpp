#include "evaluator.hpp"

#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <span>
#include <utility>
#include <optional>

#include "tensor/graph/fwd.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/graph/node.hpp"
#include "tensor/graph/view_primitive.hpp"
#include "tensor/core/tensor_spec.hpp"
#include "tensor/core/dense_size.hpp"
#include "tensor/dispatch/kernel_registry.hpp"
#include "tensor/dispatch/kernel_key.hpp"
#include "tensor/dispatch/tensor_view.hpp"
#include "tensor/backend/device_runtime.hpp"
#include "tensor/storage/materialization.hpp"
#include "tensor/storage/buffer.hpp"
#include "tensor/storage/layout.hpp"
#include "runtime_registry.hpp"

namespace minitensor::detail
{
    namespace
    {
        struct PlannedKernel
        {
            DeviceRuntime *runtime;
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
            EvaluationPlanner(const RuntimeRegistry &runtimes, const KernelRegistry &kernels) noexcept : runtimes_{runtimes}, kernels_{kernels} {}
            EvaluationPlan build(std::span<const ValueRef> roots)
            {
                for (const ValueRef &value : roots)
                {
                    visit(value);
                }
                return std::move(plan_);
            }

        private:
            void visit(const ValueRef &value)
            {
                if (!value)
                {
                    throw std::invalid_argument{"cannot evaluate an empty value"};
                }

                // cached value
                if (value->materialization() != nullptr)
                {
                    return;
                }

                const auto existing = states_.find(value.get());
                if (existing != states_.end())
                {
                    // cycle in the graph if value has already been visited
                    if (existing->second == VisitState::Visiting)
                    {
                        throw std::logic_error{"cycle in computation graph"};
                    }
                    return;
                }

                states_.emplace(value.get(), VisitState::Visiting);

                const NodeRef &node = value->producer_ref();
                // no producer and no materialization
                if (!node)
                {
                    throw std::runtime_error{"cannot evaluate unmaterialized leaf"};
                }

                // recursively visit input values
                for (const ValueRef &input : node->inputs())
                {
                    visit(input);
                }

                // add execution details to the plan
                const Primitive &primitive = node->primitive();

                // detect view operation
                if (dynamic_cast<const ViewPrimitive *>(&primitive) != nullptr)
                {
                    plan_.push_back(PlannedStep{value, node, std::nullopt});
                }
                // kernel operations
                else
                {
                    const TensorSpec &output_spec = value->spec();
                    DeviceRuntime &runtime = runtimes_.get(output_spec.device);
                    const KernelKey key{typeid(node->primitive()), output_spec.device.type(), output_spec.dtype};
                    const KernelFn &kernel = kernels_.get(key);
                    plan_.push_back(PlannedStep{value, node, PlannedKernel{&runtime, kernel}});
                }
                states_.at(value.get()) = VisitState::Finished;
            }

            const RuntimeRegistry &runtimes_;
            const KernelRegistry &kernels_;
            std::unordered_map<const Value *, VisitState> states_;
            EvaluationPlan plan_;
        };

        void execute_step(const PlannedStep &step)
        {
            // kernel operation
            if (step.kernel.has_value())
            {
                const PlannedKernel &planned_kernel = step.kernel.value();

                // construct input view
                std::vector<TensorView> input_views;
                input_views.reserve(step.node->inputs().size());

                for (const ValueRef &input : step.node->inputs())
                {
                    const Materialization *materialization = input->materialization();
                    if (materialization == nullptr)
                    {
                        throw std::logic_error{"planned operation has unmaterialized input"};
                    }
                    input_views.emplace_back(input->spec(), *materialization);
                }

                // allocate output storage
                const TensorSpec &output_spec = step.output->spec();
                BufferRef output_buffer = planned_kernel.runtime->allocate(dense_size_bytes(output_spec));
                Materialization output_materialization(std::move(output_buffer), Layout::contiguous(output_spec.shape));

                // perform operation
                {
                    MutableTensorView output_view(output_spec, output_materialization);
                    planned_kernel.function(*planned_kernel.runtime, step.node->primitive(), input_views, output_view);
                }

                // only materialize once operation succeeds
                step.output->materialize(std::move(output_materialization));
            }
            // view operation
            else
            {
                const auto *primitive = dynamic_cast<const ViewPrimitive *>(&step.node->primitive());
                if (!primitive)
                {
                    throw std::logic_error{"planned view step has non-view primitive"};
                }

                // only one input for view operations
                const ValueRef &input_value = step.node->inputs().front();
                const Materialization *materialization = input_value->materialization();
                if (materialization == nullptr)
                {
                    throw std::logic_error{"planned operation has unmaterialized input"};
                }

                Layout output_layout = primitive->derive_layout(input_value->spec(), materialization->layout(), step.output->spec());
                Materialization output_materialization = Materialization(materialization->buffer_ref(), std::move(output_layout));
                step.output->materialize(output_materialization);
            }
        }

    }

    Evaluator::Evaluator(RuntimeRegistry &runtimes, const KernelRegistry &kernels) noexcept : runtimes_{runtimes}, kernels_{kernels} {}

    void Evaluator::evaluate(std::span<const ValueRef> roots)
    {
        EvaluationPlanner planner(runtimes_, kernels_);
        EvaluationPlan plan = planner.build(roots);
        for (const PlannedStep &step : plan)
        {
            execute_step(step);
        }
    }
}