#include <minitensor/minitensor.hpp>

#include <iostream>
#include <vector>

int main()
{
    const minitensor::Tensor a = minitensor::full({2, 3}, 1.0F);
    const minitensor::Tensor b = minitensor::full({2, 3}, 2.0F);
    const minitensor::Tensor c = a + b;
    const minitensor::Shape s = c.shape();
    for (auto i = 0; i < c.rank(); ++i)
    {
        std::cout << s[i] << " ";
    }
}
