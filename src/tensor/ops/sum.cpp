#include "sum.hpp"

#include <vector>
#include <span>
#include <stdexcept>
#include <utility>
#include <memory>
#include <array>
#include <string_view>
#include <ranges>
#include <algorithm>
#include <numeric>

#include <minitensor/types.hpp>
#include <minitensor/ops.hpp>

#include "tensor/storage/layout.hpp"
#include "tensor/core/axis.hpp"
#include "tensor/graph/apply_operation.hpp"
#include "tensor/tensor_access.hpp"

namespace minitensor
{

    namespace detail
    {
        SumPrimitive::SumPrimitive(std::span<const Axis> axes, Shape::size_type input_rank, bool keep_dim) : input_rank_(input_rank), keep_dim_(keep_dim)
        {
            if (axes.size() > input_rank_)
            {
                throw std::invalid_argument{"sum cannot reduce more unique axes than the input rank"};
            }

            axes_.reserve(input_rank_);
            for (const Axis dim : axes)
            {
                axes_.push_back(normalize_axis(dim, input_rank_));
            }
            std::ranges::sort(axes_);
            if (std::ranges::adjacent_find(axes_) != axes_.end())
            {
                throw std::invalid_argument{"reduction dimensions cannot contain duplicates"};
            }
        }

        std::string_view SumPrimitive::name() const noexcept
        {
            return "sum";
        }

        TensorSpec SumPrimitive::infer(std::span<const TensorSpec> inputs) const
        {
            if (inputs.size() != 1)
            {
                throw std::invalid_argument{"sum requires a single input"};
            }

            const TensorSpec &input = inputs.front();
            if (input.shape.rank() != input_rank_)
            {
                throw std::invalid_argument{"primitive was normalized for a different input rank"};
            }

            std::vector<Extent> output_extents;

            if (keep_dim_)
            {
                // keep the same rank, but set reduced dimensions to 1
                const auto input_dimensions = input.shape.dimensions();
                output_extents.assign(input_dimensions.begin(), input_dimensions.end());
                for (const Shape::size_type axis : axes_)
                {
                    output_extents[axis] = Extent{1};
                }
            }
            else
            {
                // remove reduced dimensions
                output_extents.reserve(input.shape.rank() - axes_.size());

                auto reduced_axes_iter = axes_.begin();

                // since axes_ is sorted, we do not need a double loop
                for (Shape::size_type axis = 0; axis < input.shape.rank(); ++axis)
                {
                    if (reduced_axes_iter != axes_.end() && *reduced_axes_iter == axis)
                    {
                        ++reduced_axes_iter;
                        continue;
                    }
                    output_extents.push_back(input.shape[axis]);
                }
            }
            return TensorSpec{Shape{std::move(output_extents)}, input.dtype, input.device};
        }

        const std::vector<Shape::size_type> &SumPrimitive::axes() const noexcept
        {
            return axes_;
        }

        bool SumPrimitive::keep_dim() const noexcept
        {
            return keep_dim_;
        }

        std::vector<std::optional<Tensor>> SumPrimitive::vjp(std::span<const Tensor> inputs, const Tensor &, const Tensor &output_cotangent) const
        {
            if (inputs.size() != 1)
            {
                throw std::logic_error{"sum VJP expects 1 input"};
            }

            const Tensor &input = inputs.front();

            if (axes_.empty())
            {
                return {output_cotangent};
            }

            Tensor expanded_cotangent = output_cotangent;

            // if keep_dim was false, then we have to add the dimensions back as singletons
            if (!keep_dim_)
            {
                std::vector<Extent> expanded_dims(input.shape().dimensions().begin(), input.shape().dimensions().end());
                for (const Shape::size_type axis : axes_)
                {
                    expanded_dims[axis] = Extent{1};
                }
                expanded_cotangent = reshape(output_cotangent, Shape{std::move(expanded_dims)});
            }

            return {broadcast_to(expanded_cotangent, input.shape())};
        }

    }

    Tensor sum(const Tensor &input, std::span<const Axis> axes, bool keep_dim)
    {
        auto primitive = std::make_unique<detail::SumPrimitive>(std::move(axes), input.rank(), keep_dim);
        std::array<detail::ValueRef, 1> inputs{detail::TensorAccess::value(input)};
        detail::ValueRef output = detail::apply_operation(std::move(primitive), inputs);
        return detail::TensorAccess::make(std::move(output));
    }

    Tensor sum(const Tensor &input, bool keep_dim)
    {
        std::vector<Axis> axes(input.rank());
        std::iota(axes.begin(), axes.end(), Axis{0});
        return sum(input, axes, keep_dim);
    }

}