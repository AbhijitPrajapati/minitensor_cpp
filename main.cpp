#include <minitensor/minitensor.hpp>

#include <iostream>
#include <limits>
#include <cstdint>
#include <vector>
#include <array>
#include <span>

namespace mt = minitensor;

int main()
{
	std::array<float, 100> random_floats{ 49.45F, 60.47F, 32.87F, 29.13F, 4.9F, 45.93F, 32.3F, 16.16F, 69.88F, 26.74F, 47.43F, 55.76F, 65.83F, 55.29F, 24.71F, 57.81F, 50.35F, 18.88F, 3.68F, 17.5F, 13.03F, 63.75F, 94.74F, 6.29F, 26.97F, 22.97F,96.8F, 67.27F, 9.51F, 46.21F, 42.82F, 74.65F, 40.78F, 95.8F, 72.36F, 86.07F, 76.72F, 6.95F, 23.73F, 92.08F, 92.19F, 21.86F, 46.01F, 21.99F, 0.49F, 43.19F, 58.98F, 69.8F, 94.99F, 37.45F, 74.83F, 45.85F,96.01F, 57.12F, 3.97F, 10.72F, 56.97F, 28.0F, 52.22F, 92.42F, 48.02F, 90.7F, 14.17F, 17.64F, 89.13F, 24.84F, 42.91F, 89.8F, 75.96F, 64.45F, 8.99F, 99.1F, 94.18F, 19.56F, 60.06F, 38.49F, 25.04F, 34.46F,30.25F, 15.9F, 42.39F, 13.28F, 76.52F, 37.61F, 31.65F, 89.43F, 61.87F, 17.27F, 81.68F, 78.32F, 29.42F, 72.05F, 97.75F, 6.14F, 51.35F, 53.57F, 2.17F, 4.25F, 84.59F, 86.08F };
	mt::Tensor x = mt::from_data(random_floats, mt::Shape{ 50, 2 }) / 100.0F;
	mt::Tensor y = x * 2;

	mt::nn::Linear model(2, 2, true);

	mt::RandomKey key(54);
	auto parameters = model.initialize(key);

	std::array<float, 2> test_floats{ 0.3F, 0.6F };
	mt::Tensor test_x = mt::from_data(test_floats, mt::Shape{ 2 });
	auto pre_test_out = mt::to_vector(model(parameters, test_x));
	std::cout << "Pretest: ";
	for (auto i : pre_test_out)
	{
		std::cout << i << " ";
	}
	std::cout << "\n";


	mt::nn::SGD optimizer(0.02F);
	auto optimizer_state = optimizer.initialize(parameters);

	for (size_t epoch = 1; epoch <= 2000; ++epoch)
	{
		mt::Tensor pred = model(parameters, x);
		mt::Tensor error = pred - y;
		mt::Tensor loss = mt::sum(error * error) / 100.0F;

		if (epoch % 100 == 0)
		{
			float floss = mt::item(loss);
			std::cout << "Epoch: " << epoch << " Loss: " << floss << "\n";
		}

		auto gradients = mt::nn::parameter_gradients(loss, parameters);

		auto [new_parameters, new_state] = optimizer.step(std::move(parameters), std::move(optimizer_state), gradients);

		parameters = mt::nn::leafify_parameter_values(std::move(new_parameters));
		optimizer_state = std::move(new_state);
	}

	auto post_test_out = mt::to_vector(model(parameters, test_x));
	std::cout << "Posttest: ";
	for (auto i : post_test_out)
	{
		std::cout << i << " ";
	}
	std::cout << "\n";
}
