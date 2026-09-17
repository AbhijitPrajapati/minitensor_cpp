#include "permute.hpp"

#include <vector>
#include <span>
#include <stdexcept>
#include <utility>
#include <memory>
#include <array>
#include <string_view>
#include <optional>

#include <minitensor/types.hpp>
#include <minitensor/ops.hpp>

#include "tensor/storage/layout.hpp"
#include "tensor/core/axis.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/tensor_access.hpp"
#include "tensor/autograd/reduce_to_shape.hpp"

namespace minitensor
{

    namespace detail
    {
        PermutePrimitive::PermutePrimitive(std::span<const Axis> permutation, Shape::size_type input_rank) : input_rank_(input_rank)
        {
            if (permutation.size() != input_rank_)
            {
                throw std::invalid_argument{"permutation size does not match input rank"};
            }

            permutation_.reserve(input_rank_);
            std::vector<bool> seen(input_rank_, false);
            for (const Axis axis : permutation)
            {
                const Shape::size_type normalized = normalize_axis(axis, input_rank_);
                if (seen[normalized])
                {
                    throw std::invalid_argument{"permutation must not contain duplicate axes"};
                }
                seen[normalized] = true;
                permutation_.push_back(normalized);
            }
        }

        std::string_view PermutePrimitive::name() const noexcept
        {
            return "permute";
        }

        TensorSpec PermutePrimitive::infer(std::span<const TensorSpec> inputs) const
        {
            if (inputs.size() != 1)
            {
                throw std::invalid_argument{"permutation requires a single input"};
            }

            const TensorSpec &input = inputs.front();
            if (input.shape.rank() != input_rank_)
            {
                throw std::invalid_argument{"permutation size does not match input rank"};
            }

            std::vector<Extent> output_extents;
            output_extents.reserve(input_rank_);
            for (const auto input_axis_idx : permutation_)
            {
                output_extents.push_back(input.shape[input_axis_idx]);
            }
            return TensorSpec{Shape{std::move(output_extents)}, input.dtype, input.device};
        }

        std::optional<Layout> PermutePrimitive::try_derive_shared_layout(const TensorSpec &, const Layout &input_layout, const TensorSpec &) const
        {
            return input_layout.permuted(permutation_);
        }

        bool PermutePrimitive::requires_kernel_support() const noexcept
        {
            return false;
        }

        const std::vector<Shape::size_type> &PermutePrimitive::permutation() const noexcept
        {
            return permutation_;
        }

        std::vector<std::optional<Tensor>> PermutePrimitive::vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &output_cotangent) const
        {
            if (inputs.size() != 1)
            {
                throw std::logic_error{"permute VJP expects 1 input"};
            }

            std::vector<Axis> inverse_permutation(permutation_.size());
            for (Shape::size_type output_axis = 0; output_axis < permutation_.size(); ++output_axis)
            {
                const Shape::size_type input_axis = permutation_[output_axis];
                inverse_permutation[input_axis] = static_cast<Axis>(output_axis);
            }
            return {permute(output_cotangent, inverse_permutation)};
        }
    }

    Tensor permute(const Tensor &input, std::span<const Axis> permutation)
    {
        auto primitive = std::make_unique<detail::PermutePrimitive>(std::move(permutation), input.rank());
        std::array<detail::ValueRef, 1> inputs{detail::TensorAccess::value(input)};
        detail::ValueRef output = detail::apply_operation(std::move(primitive), inputs);
        return detail::TensorAccess::make(std::move(output));
    }

}