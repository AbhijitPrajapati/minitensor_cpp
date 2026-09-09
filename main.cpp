#include <minitensor/minitensor.hpp>

#include <iostream>
#include <limits>
#include <cstdint>
#include <vector>
#include <array>

int main()
{
    const auto f = std::array<float, 4>{ 1.0F, 5.0F, 2.0F, 3.0F };
    const minitensor::Tensor a = minitensor::from_data(f, minitensor::Shape{ 2, 2 });
    const auto g = std::array<float, 2>{ 10.0F, -3.0F };
    const minitensor::Tensor b = minitensor::from_data(g, minitensor::Shape{ 2 });
    const minitensor::Tensor c = a + b;
    const auto perm = std::array<minitensor::Axis, 2>{ 1, 0 };
    const minitensor::Tensor transposed = minitensor::permute(c, perm);
    //const minitensor::Shape s = c.shape();
    //for (auto i = 0; i < c.rank(); ++i)
    //{
    //    std::cout << s[i] << " ";
    //}
    const auto v = minitensor::to_vector(transposed);
    for (auto el : v)
    {
        std::cout << el << " ";
    }
}
