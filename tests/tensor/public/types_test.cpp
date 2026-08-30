#include <minitensor/types.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../support/test.hpp"

namespace minitensor::test
{
    void run_types_test()
    {
        const Shape scalar;
        expect(scalar.rank() == 0, "a default shape has rank zero");
        expect(scalar.is_scalar(), "a default shape represents a scalar");
        expect(scalar.dimensions().empty(), "a scalar shape has no dimensions");
        expect(scalar.numel() == 1, "a scalar shape contains one element");

        const Shape from_list{2, 3, 4};
        const std::array<Extent, 3> expected_dimensions{2, 3, 4};
        expect(from_list.rank() == expected_dimensions.size(), "initializer-list construction preserves rank");
        expect(std::ranges::equal(from_list.dimensions(), expected_dimensions),
               "initializer-list construction preserves dimensions");
        expect(from_list[1] == 3, "shape indexing returns the selected extent");
        expect(from_list.numel() == 24, "shape element count is the product of its extents");
        expect(!from_list.is_scalar(), "a ranked shape is not a scalar");

        const std::vector<Extent> dimensions{5, 2, 3};
        const Shape from_vector{dimensions};
        expect(std::ranges::equal(from_vector.dimensions(), dimensions),
               "vector construction preserves dimensions");
        expect(from_vector.numel() == 30, "vector construction computes the element count");

        const Shape empty{2, 0, std::numeric_limits<Extent>::max()};
        expect(empty.numel() == 0, "any zero extent makes the tensor empty without overflow");

        expect(Shape{2, 3} == Shape{2, 3}, "equal shapes compare equal");
        expect(Shape{2, 3} != Shape{3, 2}, "dimension order participates in shape equality");

        expect_throws<std::invalid_argument>(
            []
            {
                const Shape invalid{3, -2, 1};
                (void)invalid;
            },
            "initializer-list construction rejects negative extents");
        expect_throws<std::invalid_argument>(
            []
            {
                const Shape invalid{std::vector<Extent>{1, -1}};
                (void)invalid;
            },
            "vector construction rejects negative extents");
        expect_throws<std::overflow_error>(
            []
            {
                const Shape invalid{std::numeric_limits<Extent>::max(), 3};
                (void)invalid;
            },
            "shape construction rejects an overflowing element count");

        expect(dtype_size(DType::Float32) == sizeof(float), "Float32 reports the size of float");
        expect(dtype_name(DType::Float32) == "float32", "Float32 reports its public name");

        const auto invalid_dtype = static_cast<DType>(255);
        expect_throws<std::invalid_argument>(
            [invalid_dtype]
            {
                (void)dtype_size(invalid_dtype);
            },
            "dtype_size rejects unknown values");
        expect_throws<std::invalid_argument>(
            [invalid_dtype]
            {
                (void)dtype_name(invalid_dtype);
            },
            "dtype_name rejects unknown values");

        const Device default_device;
        expect(default_device == Device::cpu(), "a default device is CPU zero");
        expect(default_device.type() == DeviceType::Cpu, "the default device type is CPU");
        expect(default_device.index() == 0, "the default device index is zero");

        const Device indexed_cpu = Device::cpu(3);
        expect(indexed_cpu.type() == DeviceType::Cpu, "cpu constructs a CPU device");
        expect(indexed_cpu.index() == 3, "cpu preserves a nonzero device index");
        expect(indexed_cpu != default_device, "device indices participate in equality");

        const TensorOptions defaults;
        expect(defaults.dtype == DType::Float32, "tensor options default to Float32");
        expect(defaults.device == Device::cpu(), "tensor options default to CPU zero");

        const TensorOptions customized{DType::Float32, Device::cpu(4)};
        expect(customized == TensorOptions{DType::Float32, Device::cpu(4)},
               "tensor option equality compares all fields");
        expect(customized != defaults, "different tensor options compare unequal");
    }
}
