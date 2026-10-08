#pragma once

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include <minitensor/autograd.hpp>
#include <minitensor/tensor.hpp>

#include "module.hpp"
#include "parameter.hpp"

namespace minitensor::nn
{
	using ParameterGradients = std::unordered_map<ParameterId, Tensor>;

	template <typename Module>
	[[nodiscard]] ParameterGradients parameter_gradients(
		const Tensor& loss,
		const Module& module)
	{
		std::vector<ParameterId> ids;
		std::vector<Tensor> values;

		for_each_unique_parameter(module, [&](const Parameter& parameter)
			{
				if (parameter.metadata().trainable)
				{
					ids.push_back(parameter.id());
					values.push_back(parameter.value());
				}
			});
		std::vector<Tensor> gradients = grad(loss, values);

		ParameterGradients result;
		result.reserve(ids.size());

		for (std::size_t index = 0; index < ids.size(); ++index)
		{
			result.emplace(ids[index], std::move(gradients[index]));
		}

		return result;
	}
}
