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
        SumPrimitive::SumPrimitive(std::span<const Axis> axes, Shape::size_type input_rank): input_rank_(input_rank)
        {
            if (axes.size() > input_rank_)
            {
                throw std::invalid_argument{ "sum cannot reduce more unique axes than the input rank" };
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
                throw std::invalid_argument{ "sum requires a single input" };
            }

            const TensorSpec& input = inputs.front();
            if (input.shape.rank() != input_rank_)
            {
                throw std::invalid_argument{ "primitive was normalized for a different input rank" };
            }

            std::vector<Extent> output_extents;
            output_extents.reserve(input.shape.rank() - axes_.size());

            auto reduced_axes_iter = axes_.begin();

            for (Shape::size_type axis = 0; axis < input.shape.rank(); ++axis)
            {
                if (reduced_axes_iter != axes_.end() && *reduced_axes_iter == axis)
                {
                    ++reduced_axes_iter;
                    continue;
                }
                output_extents.push_back(input.shape[axis]);
            }
            return TensorSpec{ Shape{std::move(output_extents)}, input.dtype, input.device };
        }

        const std::vector<Shape::size_type>& SumPrimitive::axes() const noexcept
        {
            return axes_;
        }
    }

    Tensor sum(const Tensor& input, std::span<const Axis> axes)
    {
        auto primitive = std::make_unique<detail::SumPrimitive>(std::move(axes), input.rank());
        std::array<detail::ValueRef, 1> inputs{ detail::TensorAccess::value(input) };
        detail::ValueRef output = detail::apply_operation(std::move(primitive), inputs);
        return detail::TensorAccess::make(std::move(output));
    }

    Tensor sum(const Tensor& input)
    {
        std::vector<Axis> axes(input.rank());
        std::iota(axes.begin(), axes.end(), Axis{ 0 });
        return sum(input, axes);
    }

}