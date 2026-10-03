#pragma once

#include <optional>

#include <minitensor/nn/parameter.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::nn
{
	class Linear final
	{
	public:
		Linear(
			Extent input_features,
			Extent output_features,
			bool use_bias = true,
			TensorOptions options = {});

		[[nodiscard]] Tensor operator()(const Tensor& input) const;

		[[nodiscard]] Extent input_features() const noexcept;
		[[nodiscard]] Extent output_features() const noexcept;
		[[nodiscard]] const Parameter& weight() const noexcept;
		[[nodiscard]] const Parameter* bias() const noexcept;

	private:
		struct InitializedParameters final
		{
			Parameter weight;
			std::optional<Parameter> bias;
		};

		[[nodiscard]] static InitializedParameters initialize_parameters(
			Extent input_features,
			Extent output_features,
			bool use_bias,
			TensorOptions options);

		Linear(
			Extent input_features,
			Extent output_features,
			InitializedParameters parameters) noexcept;

		// Define parameter traversal
		template <typename Self, typename Visitor>
		static void visit_members(Self& self, Visitor& visitor)
		{
			visitor.parameter("weight", self.weight_);
			if (self.bias_)
			{
				visitor.parameter("bias", *self.bias_);
			}
		}

		friend class ModuleAccess;

		Extent input_features_;
		Extent output_features_;
		Parameter weight_;
		std::optional<Parameter> bias_;
	};
}
