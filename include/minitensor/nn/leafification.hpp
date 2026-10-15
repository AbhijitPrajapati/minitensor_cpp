#pragma once

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include <minitensor/evaluation.hpp>
#include <minitensor/nn/parameter.hpp>
#include <minitensor/nn/parameter_tree.hpp>
#include <minitensor/tensor.hpp>

namespace minitensor::nn
{
	// Parameter tensors must be leafified after each training step in order to 
	// release graphs from the previous step.

	template <typename Tree>
	[[nodiscard]] Tree leafify_parameter_values(Tree parameters)
	{
		// Collect IDs and values 
		std::vector<ParameterId> ids;
		std::vector<Tensor> values;

		for_each_unique_parameter(
			parameters, [&](const Parameter& parameter)
			{
				ids.push_back(parameter.id());
				values.push_back(parameter.value());
});

		std::vector<Tensor> leafified = leafify(values);

		// Create map
		std::unordered_map<ParameterId, Tensor> leafified_map;
		leafified_map.reserve(ids.size());

		for (std::size_t i = 0; i < ids.size(); ++i)
		{
			leafified_map.emplace(ids[i], std::move(leafified[i]));
		}

		// Transform values
		return transform_parameter_values(
		std::move(parameters),
			[&](const Parameter& parameter)
			{
				return leafified_map.at(parameter.id());
});
	}
}