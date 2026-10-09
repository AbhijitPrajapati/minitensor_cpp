#include <minitensor/evaluation.hpp>

#include <initializer_list>
#include <span>
#include <utility>
#include <vector>

#include <minitensor/tensor.hpp>

#include "graph/fwd.hpp"
#include "tensor/execution/environment.hpp"
#include "tensor/graph/leaf.hpp"
#include "tensor_access.hpp"

namespace minitensor
{
	void eval(std::span<const Tensor> tensors)
	{
		std::vector<detail::ValueRef> roots;
		roots.reserve(tensors.size());
		for (const Tensor& tensor : tensors)
		{
			roots.push_back(detail::TensorAccess::value(tensor));
		}
		detail::environment().evaluate(roots);
	}

	void eval(std::initializer_list<Tensor> tensors)
	{
		eval(std::span<const Tensor>(tensors.begin(), tensors.size()));
	}

	void eval(const Tensor& tensor)
	{
		eval({ tensor });
	}

	Tensor leafify(const Tensor& tensor)
	{
		eval(tensor);
		detail::ValueRef leaf = detail::leafify_materialized(
		detail::TensorAccess::value(tensor));
		return detail::TensorAccess::make(leaf);
	}

	std::vector<Tensor> leafify(std::span<const Tensor> tensors)
	{
		eval(tensors);

		std::vector<Tensor> output;
		output.reserve(tensors.size());
		for (const Tensor& tensor : tensors)
		{
			detail::ValueRef leaf = detail::leafify_materialized(
			detail::TensorAccess::value(tensor));
			output.push_back(detail::TensorAccess::make(std::move(leaf)));
		}
		return output;
	}

}
