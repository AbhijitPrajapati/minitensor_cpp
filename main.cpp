#include "minitensor/minitensor.hpp"

#include <iostream>
#include <vector>

int main()
{
    using minitensor::Shape;
    using minitensor::Tensor;

    auto features = Tensor::from_data(
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
        Shape{2, 3},
        true);
    const auto weights = Tensor::from_data(
        std::vector<float>{0.5F, 1.0F, 1.5F}, Shape{3});

    const auto loss = minitensor::sum(minitensor::relu(features * weights));
    loss.backward();

    std::cout << "loss = " << loss.item() << '\n';
    std::cout << "feature gradient:";
    for (const auto value : features.grad()->to_vector())
    {
        std::cout << ' ' << value;
    }
    std::cout << '\n';
}
