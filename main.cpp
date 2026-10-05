#include <minitensor/minitensor.hpp>

#include <iostream>
#include <limits>
#include <cstdint>
#include <vector>
#include <array>
#include <span>

namespace mt = minitensor;

void print_params(mt::nn::Linear& module)
{
	auto we = mt::to_vector(module.weight().value());
	auto bi = mt::to_vector(module.bias()->value());

	for (auto i = 0; i < 12; ++i)
	{
		std::cout << we[i] << " ";
	}

	std::cout << "\n";

	for (auto i = 0; i < 4; ++i)
	{
		std::cout << bi[i] << " ";
	}

	std::cout << "\n";
}

int main()
{
	auto key = mt::RandomKey(45);
	auto mod = mt::nn::Linear(3, 4, key);

	print_params(mod);

	mod = mt::nn::transform_parameter_values(
			std::move(mod),
			[&](const mt::nn::Parameter& parameter)
			{
				return mt::full_like(parameter.value(), 5.0F);
			});

	print_params(mod);
}
