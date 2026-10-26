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
    class ParameterTree final
    {
      public:
        [[nodiscard]] const ParameterId &weight() const noexcept;
        [[nodiscard]] const ParameterId *bias() const noexcept;

      private:
        ParameterTree(ParameterId weight, std::optional<ParameterId> bias) noexcept;

        template <typename Self, typename Visitor>
        static void visit_members(Self &self, Visitor &visitor)
        {
            visitor.parameter("weight", self.weight_);
            if (self.bias_)
            {
                visitor.parameter("bias", *self.bias_);
            }
        }

        ParameterId weight_;
        std::optional<ParameterId> bias_;

        friend class Linear;
        friend class ParameterTreeAccess;
    };

    Linear(Extent input_features, Extent output_features, bool use_bias = true);

    [[nodiscard]] ParameterTree initialize(ParameterBuilder &builder, RandomKey key, TensorOptions options = {}) const;

    [[nodiscard]] Tensor operator()(const ParameterMap &parameters, const ParameterTree &tree,
                                    const Tensor &input) const;

    [[nodiscard]] Extent input_features() const noexcept;
    [[nodiscard]] Extent output_features() const noexcept;
    [[nodiscard]] bool use_bias() const noexcept;

  private:
    void validate_parameters(const ParameterMap &parameters, const ParameterTree &tree) const;

    Extent input_features_;
    Extent output_features_;
    bool use_bias_;
};
} // namespace minitensor::nn
