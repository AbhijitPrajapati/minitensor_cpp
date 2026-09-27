#include <minitensor/evaluation.hpp>

#include <initializer_list>
#include <span>
#include <vector>

#include <minitensor/tensor.hpp>

#include "tensor/execution/environment.hpp"
#include "tensor/graph/fwd.hpp"
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
}
