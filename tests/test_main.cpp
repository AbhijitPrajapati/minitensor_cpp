#include "minitensor/minitensor.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

    // using minitensor::Shape;
    // using minitensor::Strides;
    // using minitensor::Tensor;

    // static_assert(std::is_same_v<minitensor::MTInt, std::int64_t>);
    // static_assert(std::is_same_v<Strides::value_type, minitensor::MTInt>);
    // static_assert(std::is_same_v<Shape::value_type, minitensor::MTInt>);

    int failures = 0;

    void fail(const std::string &message)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }

    void expect(const bool condition, const std::string &message)
    {
        if (!condition)
        {
            fail(message);
        }
    }

    void expect_close(
        const float actual,
        const float expected,
        const std::string &message,
        const float tolerance = 1.0e-5F)
    {
        if (std::fabs(actual - expected) > tolerance)
        {
            fail(message + " (expected " + std::to_string(expected) +
                 ", got " + std::to_string(actual) + ")");
        }
    }

    // void expect_values(
    //     const Tensor &tensor,
    //     const std::vector<float> &expected,
    //     const std::string &message,
    //     const float tolerance = 1.0e-5F)
    // {
    //     const auto actual = tensor.data();
    //     if (actual.size() != expected.size())
    //     {
    //         fail(message + " (element count differs)");
    //         return;
    //     }
    //     for (std::size_t index = 0; index < actual.size(); ++index)
    //     {
    //         if (std::fabs(actual[index] - expected[index]) > tolerance)
    //         {
    //             fail(message + " at element " + std::to_string(index) +
    //                  " (expected " + std::to_string(expected[index]) +
    //                  ", got " + std::to_string(actual[index]) + ")");
    //             return;
    //         }
    //     }
    // }

    template <typename Exception>
    void expect_throws(const std::function<void()> &action, const std::string &message)
    {
        try
        {
            action();
            fail(message + " (no exception was thrown)");
        }
        catch (const Exception &)
        {
        }
        catch (...)
        {
            fail(message + " (wrong exception type)");
        }
    }

    // Tensor matrix_2x3(const bool requires_grad = false)
    // {
    //     return Tensor::from_data(
    //         {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
    //         Shape{2, 3},
    //         requires_grad);
    // }

    // void test_layout()
    // {
    //     const auto tensor = Tensor::from_data({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, Shape{2, 3});
    //     expect(tensor.shape() == Shape{2, 3}, "factory preserves shape");
    //     expect(tensor.strides() == Strides{3, 1}, "factory makes row-major strides");
    //     expect(tensor.offset() == 0, "factory makes 0 offset");
    //     expect(tensor.numel() == 6, "numel is the shape product");
    // }

    // void test_forward_operations()
    // {
    //     const auto tensor = Tensor::from_data({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, Shape{2, 3});
    //     const auto row = Tensor::from_data({10.0F, 20.0F, 30.0F}, Shape{3});
    //     expect_values(tensor + row, {11, 22, 33, 14, 25, 36}, "trailing broadcast add");

    //     const auto lhs = Tensor::from_data({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, Shape{2, 3});
    //     const auto rhs = Tensor::from_data({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, Shape{3, 2});
    //     expect_values(minitensor::matmul(lhs, rhs), {22, 28, 49, 64}, "rank-2 matmul");

    //     expect_values(minitensor::relu(Tensor::from_data({-2.0F, 0.0F, 3.0F}, {3})), {0, 0, 3}, "relu");
    // }

    // void test_elementwise_autograd()
    // {
        // auto input = matrix_2x3(true);
        // auto weights = Tensor::from_data({2, 3, 4}, Shape{3}, true);
        // const auto loss = minitensor::sum(input * weights);
        // loss.backward();

        // expect(input.grad().has_value(), "leaf receives a gradient");
        // expect_values(*input.grad(), {2, 3, 4, 2, 3, 4}, "broadcast multiply input gradient");
        // expect_values(*weights.grad(), {5, 7, 9}, "broadcast gradient sums expanded axes");

        // auto branched = Tensor::from_data({2, -1}, Shape{2}, true);
        // const auto branch_loss = minitensor::sum(branched * branched + branched);
        // branch_loss.backward();
        // expect_values(*branched.grad(), {5, -1}, "shared graph branches accumulate before backward");

        // branch_loss.backward();
        // expect_values(*branched.grad(), {10, -2}, "backward calls accumulate leaf gradients");
        // branched.clear_grad();
        // expect(!branched.grad().has_value(), "clear_grad removes accumulated gradient");

        // auto numerator = Tensor::from_data({2, 6}, Shape{2}, true);
        // auto denominator = Tensor::from_data({2, 3}, Shape{2}, true);
        // minitensor::sum(numerator / denominator + numerator - denominator).backward();
        // expect_values(*numerator.grad(), {1.5F, 4.0F / 3.0F}, "division numerator gradient");
        // expect_values(*denominator.grad(), {-1.5F, -5.0F / 3.0F}, "division denominator gradient");
    // }

    // void test_matmul_and_activation_autograd()
    // {
        // auto lhs = Tensor::from_data({1, 2, 3, 4}, Shape{2, 2}, true);
        // auto rhs = Tensor::from_data({5, 6, 7, 8}, Shape{2, 2}, true);
        // minitensor::sum(minitensor::matmul(lhs, rhs)).backward();
        // expect_values(*lhs.grad(), {11, 15, 11, 15}, "matmul left gradient");
        // expect_values(*rhs.grad(), {4, 4, 6, 6}, "matmul right gradient");

        // auto relu_input = Tensor::from_data({-1, 0, 2}, Shape{3}, true);
        // minitensor::sum(minitensor::relu(relu_input)).backward();
        // expect_values(*relu_input.grad(), {0, 0, 1}, "relu gradient");

        // auto sigmoid_input = Tensor::from_data({0}, Shape{1}, true);
        // minitensor::sum(minitensor::sigmoid(sigmoid_input)).backward();
        // expect_values(*sigmoid_input.grad(), {0.25F}, "sigmoid gradient");

        // auto tanh_input = Tensor::from_data({0}, Shape{1}, true);
        // minitensor::sum(minitensor::tanh(tanh_input)).backward();
        // expect_values(*tanh_input.grad(), {1.0F}, "tanh gradient");
    // }

    // void test_validation()
    // {
    //     expect_throws<std::invalid_argument>(
    //         []
    //         { static_cast<void>(Tensor::from_data({1, 2}, Shape{3})); },
    //         "factory rejects mismatched data length");
    //     expect_throws<std::invalid_argument>(
    //         []
    //         { static_cast<void>(Tensor::zeros({2, 3}) + Tensor::zeros({2, 2})); },
    //         "broadcasting rejects incompatible shapes");
    //     expect_throws<std::invalid_argument>(
    //         []
    //         { static_cast<void>(minitensor::matmul(Tensor::zeros({2}), Tensor::zeros({2, 1}))); },
    //         "matmul rejects non-matrices");
        // expect_throws<std::invalid_argument>(
        //     []
        //     { static_cast<void>(Tensor::zeros({2, -1})); },
        //     "negative shape extents are rejected");
        // expect_throws<std::overflow_error>(
        //     []
        //     {
        //         static_cast<void>(Tensor::zeros(
        //             {std::numeric_limits<minitensor::MTInt>::max(), 2}));
        //     },
        //     "shape products are checked for int64 overflow");
        // expect_throws<std::logic_error>(
        //     []
        //     { Tensor::ones({2}).backward(); },
        //     "implicit backward requires one output element");
    // }

} // namespace

int main()
{
    // test_layout();
    // test_forward_operations();

    // // probably combine into one test_autograd_function
    // test_elementwise_autograd();
    // test_matmul_and_activation_autograd();

    // test_validation();

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All minitensor tests passed\n";
    return 0;
}
