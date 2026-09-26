#pragma once

#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <minitensor/tensor.hpp>

#include "tensor/core/tensor_spec.hpp"
#include "tensor/graph/primitive.hpp"
#include "tensor/storage/layout.hpp"

namespace minitensor::test
{
	class IdentitySpecPrimitive final : public detail::Primitive
	{
	public:
		[[nodiscard]] std::string_view name() const noexcept override
		{
			return "test_identity";
		}

		[[nodiscard]] detail::TensorSpec infer(std::span<const detail::TensorSpec> inputs) const override
		{
			if (inputs.size() != 1)
			{
				throw std::invalid_argument{ "expected one input" };
			}
			return inputs.front();
		}
	};

	class DestructionTrackedPrimitive final : public detail::Primitive
	{
	public:
		explicit DestructionTrackedPrimitive(bool& destroyed) noexcept : destroyed_{ destroyed }
		{
			destroyed_ = false;
		}

		~DestructionTrackedPrimitive() override
		{
			destroyed_ = true;
		}

		[[nodiscard]] std::string_view name() const noexcept override
		{
			return "test_destruction_tracked_identity";
		}

		[[nodiscard]] detail::TensorSpec infer(std::span<const detail::TensorSpec> inputs) const override
		{
			if (inputs.size() != 1)
			{
				throw std::invalid_argument{ "expected one input" };
			}
			return inputs.front();
		}

	private:
		bool& destroyed_;
	};

	class IdentityStorageSharingPrimitive final : public detail::Primitive
	{
	public:
		[[nodiscard]] std::string_view name() const noexcept override
		{
			return "test_identity_storage_sharing";
		}

		[[nodiscard]] detail::TensorSpec infer(std::span<const detail::TensorSpec> inputs) const override
		{
			if (inputs.size() != 1)
			{
				throw std::invalid_argument{ "expected one input" };
			}

			return inputs.front();
		}

		[[nodiscard]] std::optional<detail::Layout> try_derive_shared_layout(
			const detail::TensorSpec&,
			const detail::Layout& input_layout,
			const detail::TensorSpec&) const override
		{
			return input_layout;
		}

		[[nodiscard]] bool requires_kernel_support() const noexcept override
		{
			return false;
		}
	};

	class RuleBasedVjpPrimitive final : public detail::Primitive
	{
	public:
		using Rule = std::function<std::vector<std::optional<Tensor>>(
			std::span<const Tensor>, const Tensor&, const Tensor&)>;

		RuleBasedVjpPrimitive(
			std::size_t input_count,
			detail::TensorSpec output_spec,
			Rule rule)
			: input_count_{ input_count },
			output_spec_{ std::move(output_spec) },
			rule_{ std::move(rule) }
		{
			if (!rule_)
			{
				throw std::invalid_argument{ "test VJP rule cannot be empty" };
			}
		}

		[[nodiscard]] std::string_view name() const noexcept override
		{
			return "test_rule_based_vjp";
		}

		[[nodiscard]] detail::TensorSpec infer(
			std::span<const detail::TensorSpec> inputs) const override
		{
			if (inputs.size() != input_count_)
			{
				throw std::invalid_argument{ "unexpected test VJP input count" };
			}
			return output_spec_;
		}

		[[nodiscard]] std::vector<std::optional<Tensor>> vjp(
			std::span<const Tensor> inputs,
			const Tensor& output,
			const Tensor& output_cotangent) const override
		{
			return rule_(inputs, output, output_cotangent);
		}

	private:
		std::size_t input_count_;
		detail::TensorSpec output_spec_;
		Rule rule_;
	};
}
