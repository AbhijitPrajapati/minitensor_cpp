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

    using minitensor::Shape;
    using minitensor::Strides;
    using minitensor::Tensor;

    static_assert(std::is_same_v<minitensor::Index, std::int64_t>);
    static_assert(std::is_same_v<Shape::value_type, minitensor::Index>);
    static_assert(std::is_same_v<Strides::value_type, minitensor::Index>);

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

    void expect_values(
        const Tensor &tensor,
        const std::vector<float> &expected,
        const std::string &message,
        const float tolerance = 1.0e-5F)
    {
        const auto actual = tensor.to_vector();
        if (actual.size() != expected.size())
        {
            fail(message + " (element count differs)");
            return;
        }
        for (std::size_t index = 0; index < actual.size(); ++index)
        {
            if (std::fabs(actual[index] - expected[index]) > tolerance)
            {
                fail(message + " at element " + std::to_string(index) +
                     " (expected " + std::to_string(expected[index]) +
                     ", got " + std::to_string(actual[index]) + ")");
                return;
            }
        }
    }

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

    Tensor matrix_2x3(const bool requires_grad = false)
    {
        return Tensor::from_data(
            {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
            Shape{2, 3},
            requires_grad);
    }

    void test_layout_and_views()
    {
        const auto tensor = matrix_2x3();
        expect(tensor.shape() == Shape{2, 3}, "factory preserves shape");
        expect(tensor.strides() == Strides{3, 1}, "factory makes row-major strides");
        expect(tensor.numel() == 6, "numel is the shape product");
        expect(tensor.is_contiguous(), "fresh tensor is contiguous");

        const auto transposed = tensor.transpose(0, 1);
        expect(transposed.shape() == Shape{3, 2}, "transpose swaps shape dimensions");
        expect(transposed.strides() == Strides{1, 3}, "transpose swaps strides");
        expect(!transposed.is_contiguous(), "ordinary transpose is non-contiguous");
        expect_values(transposed, {1, 4, 2, 5, 3, 6}, "transpose logical values");

        const auto sliced = tensor.slice(1, 1, 3);
        expect(sliced.shape() == Shape{2, 2}, "slice computes output extent");
        expect(sliced.storage_offset() == 1, "slice adjusts storage offset");
        expect_values(sliced, {2, 3, 5, 6}, "slice logical values");

        const auto stepped = tensor.slice(1, 0, 3, 2);
        expect(stepped.strides() == Strides{3, 2}, "step scales selected stride");
        expect_values(stepped, {1, 3, 4, 6}, "stepped slice values");

        const auto dense = transposed.contiguous();
        expect(dense.is_contiguous(), "contiguous materializes a dense tensor");
        expect_values(dense, {1, 4, 2, 5, 3, 6}, "contiguous preserves logical order");

        const auto reshaped = transposed.reshape({2, 3});
        expect(reshaped.shape() == Shape{2, 3}, "reshape changes shape");
        expect(reshaped.is_contiguous(), "non-contiguous reshape materializes first");
        expect_values(reshaped, {1, 4, 2, 5, 3, 6}, "reshape preserves logical order");
    }

    void test_forward_operations()
    {
        const auto tensor = matrix_2x3();
        const auto row = Tensor::from_data({10.0F, 20.0F, 30.0F}, Shape{3});
        expect_values(tensor + row, {11, 22, 33, 14, 25, 36}, "trailing broadcast add");
        expect_values(tensor * 2.0F - 1.0F, {1, 3, 5, 7, 9, 11}, "scalar arithmetic");
        expect_values(12.0F / tensor, {12, 6, 4, 3, 2.4F, 2}, "reverse scalar division");

        const auto lhs = Tensor::from_data({1, 2, 3, 4, 5, 6}, Shape{2, 3});
        const auto rhs = Tensor::from_data({1, 2, 3, 4, 5, 6}, Shape{3, 2});
        expect_values(minitensor::matmul(lhs, rhs), {22, 28, 49, 64}, "rank-2 matmul");
        expect_values(
            minitensor::matmul(lhs.transpose(0, 1), Tensor::ones({2, 1})),
            {5, 7, 9},
            "matmul accepts strided inputs");
        expect_values(
            minitensor::matmul(Tensor::zeros({2, 0}), Tensor::zeros({0, 3})),
            {0, 0, 0, 0, 0, 0},
            "matmul initializes outputs with an empty contraction dimension");

        expect_close(minitensor::sum(tensor).item(), 21.0F, "sum all elements");
        expect_values(minitensor::sum(tensor, 0), {5, 7, 9}, "sum along dimension zero");
        expect_values(minitensor::sum(tensor, 1, true), {6, 15}, "sum keepdim values");
        expect(minitensor::sum(tensor, 1, true).shape() == Shape{2, 1}, "sum keepdim shape");
        expect_values(
            minitensor::sum(tensor.transpose(0, 1), 1),
            {5, 7, 9},
            "sum accepts a strided input layout");

        expect_values(minitensor::relu(Tensor::from_data({-2, 0, 3}, {3})), {0, 0, 3}, "relu");
        expect_values(minitensor::sigmoid(Tensor::from_data({0}, {1})), {0.5F}, "sigmoid");
        expect_values(minitensor::tanh(Tensor::from_data({0}, {1})), {0.0F}, "tanh");
        expect_close(minitensor::sum(Tensor::zeros({0, 3})).item(), 0.0F, "empty sum is zero");
    }

    void test_elementwise_autograd()
    {
        auto input = matrix_2x3(true);
        auto weights = Tensor::from_data({2, 3, 4}, Shape{3}, true);
        const auto loss = minitensor::sum(input * weights);
        loss.backward();

        expect(input.grad().has_value(), "leaf receives a gradient");
        expect_values(*input.grad(), {2, 3, 4, 2, 3, 4}, "broadcast multiply input gradient");
        expect_values(*weights.grad(), {5, 7, 9}, "broadcast gradient sums expanded axes");

        auto branched = Tensor::from_data({2, -1}, Shape{2}, true);
        const auto branch_loss = minitensor::sum(branched * branched + branched);
        branch_loss.backward();
        expect_values(*branched.grad(), {5, -1}, "shared graph branches accumulate before backward");

        branch_loss.backward();
        expect_values(*branched.grad(), {10, -2}, "backward calls accumulate leaf gradients");
        branched.clear_grad();
        expect(!branched.grad().has_value(), "clear_grad removes accumulated gradient");

        auto numerator = Tensor::from_data({2, 6}, Shape{2}, true);
        auto denominator = Tensor::from_data({2, 3}, Shape{2}, true);
        minitensor::sum(numerator / denominator + numerator - denominator).backward();
        expect_values(*numerator.grad(), {1.5F, 4.0F / 3.0F}, "division numerator gradient");
        expect_values(*denominator.grad(), {-1.5F, -5.0F / 3.0F}, "division denominator gradient");
    }

    void test_view_and_reduction_autograd()
    {
        auto input = matrix_2x3(true);
        minitensor::sum(input.slice(1, 0, 3, 2)).backward();
        expect_values(*input.grad(), {1, 0, 1, 1, 0, 1}, "slice backward scatters into input shape");

        auto empty_slice_input = matrix_2x3(true);
        minitensor::sum(empty_slice_input.slice(1, 1, 1, 2)).backward();
        expect_values(
            *empty_slice_input.grad(),
            {0, 0, 0, 0, 0, 0},
            "empty slice backward preserves a zero destination");

        auto transposed_input = matrix_2x3(true);
        const auto weights = Tensor::from_data({1, 2, 3, 4, 5, 6}, Shape{2, 3});
        const auto loss = minitensor::sum(transposed_input.transpose(0, 1).reshape({2, 3}) * weights);
        loss.backward();
        expect_values(
            *transposed_input.grad(),
            {1, 3, 5, 2, 4, 6},
            "transpose-contiguous-reshape backward preserves logical mapping");

        auto reduction_input = matrix_2x3(true);
        const auto reduction = minitensor::sum(reduction_input, 0);
        reduction.backward(Tensor::from_data({1, 2, 3}, Shape{3}));
        expect_values(
            *reduction_input.grad(),
            {1, 2, 3, 1, 2, 3},
            "dimension sum backward broadcasts supplied gradient");

        auto direct_transpose_input = matrix_2x3(true);
        direct_transpose_input.transpose(0, 1).backward(
            Tensor::from_data({1, 2, 3, 4, 5, 6}, Shape{3, 2}));
        expect_values(
            *direct_transpose_input.grad(),
            {1, 3, 5, 2, 4, 6},
            "transpose backward inverts the permutation");
        expect(direct_transpose_input.grad()->is_contiguous(), "stored leaf gradients are contiguous");
    }

    void test_matmul_and_activation_autograd()
    {
        auto lhs = Tensor::from_data({1, 2, 3, 4}, Shape{2, 2}, true);
        auto rhs = Tensor::from_data({5, 6, 7, 8}, Shape{2, 2}, true);
        minitensor::sum(minitensor::matmul(lhs, rhs)).backward();
        expect_values(*lhs.grad(), {11, 15, 11, 15}, "matmul left gradient");
        expect_values(*rhs.grad(), {4, 4, 6, 6}, "matmul right gradient");

        auto relu_input = Tensor::from_data({-1, 0, 2}, Shape{3}, true);
        minitensor::sum(minitensor::relu(relu_input)).backward();
        expect_values(*relu_input.grad(), {0, 0, 1}, "relu gradient");

        auto sigmoid_input = Tensor::from_data({0}, Shape{1}, true);
        minitensor::sum(minitensor::sigmoid(sigmoid_input)).backward();
        expect_values(*sigmoid_input.grad(), {0.25F}, "sigmoid gradient");

        auto tanh_input = Tensor::from_data({0}, Shape{1}, true);
        minitensor::sum(minitensor::tanh(tanh_input)).backward();
        expect_values(*tanh_input.grad(), {1.0F}, "tanh gradient");
    }

    void test_validation()
    {
        expect_throws<std::invalid_argument>(
            []
            { static_cast<void>(Tensor::from_data({1, 2}, Shape{3})); },
            "factory rejects mismatched data length");
        expect_throws<std::invalid_argument>(
            []
            { static_cast<void>(Tensor::zeros({2, 3}) + Tensor::zeros({2, 2})); },
            "broadcasting rejects incompatible shapes");
        expect_throws<std::invalid_argument>(
            []
            { static_cast<void>(minitensor::matmul(Tensor::zeros({2}), Tensor::zeros({2, 1}))); },
            "matmul rejects non-matrices");
        expect_throws<std::out_of_range>(
            []
            { static_cast<void>(Tensor::zeros({2, 3}).slice(1, 0, 4)); },
            "slice validates bounds");
        expect_throws<std::invalid_argument>(
            []
            { static_cast<void>(Tensor::zeros({2, -1})); },
            "negative shape extents are rejected");
        expect_throws<std::overflow_error>(
            []
            {
                static_cast<void>(Tensor::zeros(
                    {std::numeric_limits<minitensor::Index>::max(), 2}));
            },
            "shape products are checked for int64 overflow");
        expect_throws<std::out_of_range>(
            []
            { static_cast<void>(Tensor::zeros({2, 3}).transpose(-1, 0)); },
            "negative transpose dimensions are rejected");
        expect_throws<std::out_of_range>(
            []
            { static_cast<void>(Tensor::zeros({2, 3}).slice(1, -1, 2)); },
            "negative slice indices are rejected");
        expect_throws<std::out_of_range>(
            []
            { static_cast<void>(minitensor::sum(Tensor::zeros({2, 3}), -1)); },
            "negative reduction dimensions are rejected");
        expect_throws<std::logic_error>(
            []
            { Tensor::ones({2}).backward(); },
            "implicit backward requires one output element");
    }

} // namespace

int main()
{
    test_layout_and_views();
    test_forward_operations();
    test_elementwise_autograd();
    test_view_and_reduction_autograd();
    test_matmul_and_activation_autograd();
    test_validation();

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All minitensor tests passed\n";
    return 0;
}
