#include "engine.hpp"

#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <minitensor/ops/creation.hpp>
#include <minitensor/ops/elementwise.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/fwd.hpp"
#include "tensor/graph/node.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/tensor_access.hpp"

namespace minitensor::detail
{
	namespace
	{
		struct ReverseStep final
		{
			ValueRef output;
			NodeRef node;
			std::vector<std::size_t> input_indices_needed;
		};

		using ReversePlan = std::vector<ReverseStep>;

		enum class VisitState
		{
			Visiting,
			Finished
		};

		struct VisitRecord final
		{
			VisitState state;
			bool depends_on_target;
		};

		class ReversePlanner final
		{
		public:
			explicit ReversePlanner(std::span<const ValueRef> targets)
			{
				// populate targets
				targets_.reserve(targets.size());
				for (const ValueRef& target : targets)
				{
					if (!target)
					{
						throw std::invalid_argument{ "reverse plan target cannot be null" };
					}
					targets_.insert(target.get());
				}
			}

			[[nodiscard]] ReversePlan build(const ValueRef& output)
			{
				if (!output)
				{
					throw std::invalid_argument{ "reverse plan output cannot be null" };
				}

				if (targets_.empty())
				{
					return {};
				}

				// visit preceding nodes
				(void)visit(output);
				return std::move(plan_);
			}

		private:
			// returns whether value is dependent on a target or is a target
			[[nodiscard]] bool visit(const ValueRef& value)
			{
				if (!value)
				{
					throw std::invalid_argument{ "reverse planner cannot visit a null value" };
				}

				const Value* const key = value.get();
				const auto existing = visits_.find(key);
				if (existing != visits_.end())
				{
					// cycle in the graph if value has already been visited
					if (existing->second.state == VisitState::Visiting)
					{
						throw std::logic_error{ "cycle in computation graph" };
					}
					return existing->second.depends_on_target;
				}

				visits_.emplace(key, VisitRecord{ VisitState::Visiting, false });

				const bool is_target = targets_.contains(key);
				std::vector<std::size_t> input_indices_needed;

				const NodeRef& node = value->producer_ref();
				if (node)
				{
					const std::span<const ValueRef> inputs = node->inputs();
					input_indices_needed.reserve(inputs.size());

					// record input indices required for differentiation
					for (std::size_t input_index = 0; input_index < inputs.size(); ++input_index)
					{
						if (visit(inputs[input_index]))
						{
							input_indices_needed.push_back(input_index);
						}
					}
				}

				// either is target or depends on target directly or indirectly
				const bool depends_on_target = is_target || !input_indices_needed.empty();

				if (!input_indices_needed.empty())
				{
					plan_.push_back(ReverseStep{ value, node, std::move(input_indices_needed) });
				}

				VisitRecord& record = visits_.at(key);
				record.state = VisitState::Finished;
				record.depends_on_target = depends_on_target;
				return depends_on_target;
			}

			std::unordered_set<const Value*> targets_;
			std::unordered_map<const Value*, VisitRecord> visits_;
			ReversePlan plan_; // accumulated in topological order, then iterated in reverse
		};
	}

	std::vector<Tensor> reverse_vjp(const ValueRef& output, std::span<const ValueRef> targets, const Tensor& output_cotangent)
	{
		if (!output)
		{
			throw std::invalid_argument("reverse vjp output cannot be null");
		}

		// validate cotangent
		const TensorSpec& output_spec = output->spec();
		if (output_cotangent.device() != output_spec.device)
		{
			throw std::invalid_argument{ "output cotangent does not match output device" };
		}
		if (output_cotangent.dtype() != output_spec.dtype)
		{
			throw std::invalid_argument{ "output cotangent does not match output dtype" };
		}
		if (output_cotangent.shape() != output_spec.shape)
		{
			throw std::invalid_argument{ "output cotangent does not match output shape" };
		}

		// create plan
		ReversePlanner planner(targets);
		ReversePlan plan = planner.build(output);

		// seed initial cotangent
		std::unordered_map<const Value*, Tensor> cotangents;
		cotangents.emplace(output.get(), output_cotangent);

		// propagate in reverse
		for (auto step = plan.rbegin(); step != plan.rend(); ++step)
		{
			// find step's output cotangent
			auto output_cotangent_it = cotangents.find(step->output.get());
			if (output_cotangent_it == cotangents.end())
			{
				continue;
			}
			Tensor step_output_cotangent = output_cotangent_it->second;

			// get input and output tensors
			const std::span<const ValueRef> inputs = step->node->inputs();
			std::vector<Tensor> input_tensors;
			input_tensors.reserve(inputs.size());
			for (const ValueRef& input : inputs)
			{
				input_tensors.push_back(TensorAccess::make(input));
			}

			Tensor step_output = TensorAccess::make(step->output);

			// run vjp rule
			std::vector<std::optional<Tensor>> contributions =
				step->node->primitive().vjp(
					input_tensors,
					step_output,
					step_output_cotangent);

			if (contributions.size() != inputs.size())
			{
				throw std::logic_error{ "primitive VJP returned a different number of input cotangents" };
			}

			// account for all contributions
			for (std::size_t input_index : step->input_indices_needed)
			{
				std::optional<Tensor>& contribution = contributions[input_index];

				// null opt = no propagation
				if (!contribution.has_value())
				{
					continue;
				}

				const ValueRef& input = inputs[input_index];
				const TensorSpec& input_spec = input->spec();
				Tensor input_cotangent = std::move(*contribution);

				// specs must match
				if (input_cotangent.shape() != input_spec.shape ||
					input_cotangent.dtype() != input_spec.dtype ||
					input_cotangent.device() != input_spec.device)
				{
					throw std::logic_error(
						"primitive VJP returned an input cotangent with an "
						"incorrect tensor specification");
				}

				// add contribution if exising, emplace if not
				auto existing = cotangents.find(input.get());
				if (existing == cotangents.end())
				{
					cotangents.emplace(input.get(), std::move(input_cotangent));
				}
				else
				{
					existing->second = existing->second + input_cotangent;
				}
			}
		}

		// collect target cotangents
		std::vector<Tensor> target_cotangents;
		target_cotangents.reserve(targets.size());
		for (const ValueRef& value : targets)
		{
			auto cotangent = cotangents.find(value.get());
			if (cotangent != cotangents.end())
			{
				target_cotangents.push_back(cotangent->second);
				continue;
			}

			// use 0 tensor if no cotangent
			const TensorSpec& spec = value->spec();
			target_cotangents.push_back(full(spec.shape, 0.0F, TensorOptions{ spec.dtype, spec.device }));
		}
		return target_cotangents;
	}
}
