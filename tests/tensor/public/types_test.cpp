#include <minitensor/types.hpp>

namespace minitensor::test
{
    void run_types_test()
    {
        Shape{}.numel() == 1;
        Shape{}.is_scalar();
        Shape{}.rank() == 0;
        Shape{2, 3, 4}.dimensions() == {2, 3, 4};
        Shape{2, 3, 4}.numel() == 24;
        Shape{2, 0, 3}.numel() == 0;
        Shape{3, -2, 1}; // shouldnt work
        Shape{2, 3} == Shape{2, 3};
        Shape{2, 3} != Shape{3, 2};
        Shape{1, 2, 3}[2] == 3;

        dtype_size(DType::Float32) == sizeof(float);
        dtype_name(DType::Float32) == "float32";

        Device::cpu().index() == 0;
        Device::cpu().type() == DeviceType::Cpu;

        TensorOptions{}.dtype == DType::Float32;
        TensorOptions{}.device == Device::cpu();
    }
}
