#pragma once

#include <optional>

#include <minitensor/nn/parameter.hpp>
#include <minitensor/random.hpp>
#include <minitensor/tensor.hpp>
#include <minitensor/types.hpp>

namespace minitensor::nn
{
	class Linear final
	{
	public:
		class Parameters final
		{
		public:
			[[nodiscard]] const Parameter& weight() const noexcept;
			[[nodiscard]] const Parameter* bias() const noexcept;

		private:
			Parameters(Parameter weight, std::optional<Parameter> bias) noexcept;

			template <typename Self, typename Visitor>
			static void visit_members(Self& self, Visitor& visitor)
			{
				visitor.parameter("weight", self.weight_);
				if (self.bias_)
				{
					visitor.parameter("bias", *self.bias_);
				}
			}

			Parameter weight_;
			std::optional<Parameter> bias_;

			friend class Linear;
			friend class ParameterTreeAccess;
		};

		Linear(
			Extent input_features,
			Extent output_features,
			bool use_bias = true);

		[[nodiscard]] Parameters initialize(
			RandomKey key,
			TensorOptions options = {}) const;

		[[nodiscard]] Tensor operator()(
			const Parameters& parameters,
			const Tensor& input) const;

		[[nodiscard]] Extent input_features() const noexcept;
		[[nodiscard]] Extent output_features() const noexcept;
		[[nodiscard]] bool use_bias() const noexcept;

	private:
		Extent input_features_;
		Extent output_features_;
		bool use_bias_;
	};
}
