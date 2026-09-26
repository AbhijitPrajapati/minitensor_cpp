#include <minitensor/ops.hpp>

#include <array>
#include <span>
#include <stdexcept>
#include <utility>

#include "../support/test.hpp"

namespace minitensor::test
{
	void run_operations_test()
	{
		const Tensor matrix = full(Shape{ 2, 3 }, 4.25F);
		expect(matrix.shape() == Shape{ 2, 3 }, "full exposes its requested shape");
		expect(matrix.rank() == 2, "a tensor handle reports its rank");
		expect(matrix.numel() == 6, "a tensor handle reports its element count");
		expect(matrix.dtype() == DType::Float32, "full uses the default dtype");
		expect(matrix.device() == Device::cpu(), "full uses the default device");

		const Tensor scalar = full(Shape{}, -1.0F);
		expect(scalar.shape().is_scalar(), "full constructs scalar tensors");
		expect(scalar.rank() == 0, "a scalar tensor handle has rank zero");
		expect(scalar.numel() == 1, "a scalar tensor handle has one element");

		const Tensor empty = full(Shape{ 2, 0, 3 }, 0.0F);
		expect(empty.numel() == 0, "full preserves empty tensor shapes");

		const TensorOptions options{ DType::Float32, Device::cpu(7) };
		const Tensor configured = full(Shape{ 3 }, 2.0F, options);
		expect(configured.dtype() == options.dtype, "full preserves an explicit dtype option");
		expect(configured.device() == options.device, "full preserves an explicit device option");

		const Tensor zero_tensor = zeros(Shape{ 2, 3 });
		const Tensor one_tensor = ones(Shape{ 2, 3 });
		expect(zero_tensor.shape() == Shape{ 2, 3 } && one_tensor.shape() == Shape{ 2, 3 },
			   "zeros and ones preserve their requested shapes");

		const Tensor like_configured = full_like(configured, -2.0F);
		expect(like_configured.shape() == configured.shape() &&
				   like_configured.dtype() == configured.dtype() &&
				   like_configured.device() == configured.device(),
			   "full_like inherits shape, dtype, and device when options are omitted");
		expect(zeros_like(configured).device() == configured.device() &&
				   ones_like(configured).device() == configured.device(),
			   "zeros_like and ones_like inherit the input device");
		expect(zeros_like(configured, TensorOptions{}).device() == Device::cpu(),
			   "explicit like-options override the input options");

		Tensor copied = matrix;
		expect(copied.shape() == matrix.shape(), "copy construction preserves tensor metadata");
		Tensor assigned = full(Shape{ 1 }, 0.0F);
		assigned = matrix;
		expect(assigned.shape() == matrix.shape(), "copy assignment replaces the tensor handle");
		const Tensor moved{ std::move(copied) };
		expect(moved.shape() == matrix.shape(), "move construction transfers the tensor handle");

		const Tensor same_shape_sum = matrix + full(Shape{ 2, 3 }, 1.0F);
		expect(same_shape_sum.shape() == Shape{ 2, 3 }, "addition preserves equal input shapes");
		expect(same_shape_sum.dtype() == matrix.dtype(), "addition preserves the input dtype");
		expect(same_shape_sum.device() == matrix.device(), "addition preserves the input device");

		const Tensor broadcast_sum = full(Shape{ 2, 1, 4 }, 1.0F) + full(Shape{ 3, 4 }, 2.0F);
		expect(broadcast_sum.shape() == Shape{ 2, 3, 4 },
			   "addition broadcasts singleton and missing leading dimensions");

		const Tensor scalar_sum = scalar + full(Shape{ 2, 3 }, 1.0F);
		expect(scalar_sum.shape() == Shape{ 2, 3 }, "addition broadcasts a scalar to a ranked tensor");

		const Tensor configured_scalar_result = 2.0F - configured;
		expect(configured_scalar_result.shape() == configured.shape() &&
				   configured_scalar_result.dtype() == configured.dtype() &&
				   configured_scalar_result.device() == configured.device(),
			   "scalar overloads preserve the tensor specification");

		const Tensor empty_sum = full(Shape{ 2, 0, 3 }, 1.0F) + full(Shape{ 1, 3 }, 2.0F);
		expect(empty_sum.shape() == Shape{ 2, 0, 3 }, "addition broadcasts compatible empty shapes");
		expect(empty_sum.numel() == 0, "a broadcasted empty result remains empty");

		const Tensor negated = -matrix;
		expect(negated.shape() == matrix.shape() &&
				   negated.dtype() == matrix.dtype() &&
				   negated.device() == matrix.device(),
			   "negation preserves the input shape, dtype, and device");

		expect(exp(matrix).shape() == matrix.shape() &&
				   log(matrix).shape() == matrix.shape() &&
				   sqrt(matrix).shape() == matrix.shape() &&
				   tanh(matrix).shape() == matrix.shape(),
			   "unary math operations preserve the input tensor specification");

		const Tensor broadcast_difference = full(Shape{ 2, 1 }, 2.0F) - full(Shape{ 1, 3 }, 3.0F);
		expect(broadcast_difference.shape() == Shape{ 2, 3 },
			   "subtraction broadcasts compatible input shapes");

		const Tensor broadcast_product = full(Shape{ 2, 1 }, 2.0F) * full(Shape{ 1, 3 }, 3.0F);
		expect(broadcast_product.shape() == Shape{ 2, 3 },
			   "multiplication broadcasts compatible input shapes");
		expect(broadcast_product.dtype() == DType::Float32 &&
				   broadcast_product.device() == Device::cpu(),
			   "multiplication preserves the input dtype and device");

		const Tensor broadcast_quotient = full(Shape{ 2, 1 }, 2.0F) / full(Shape{ 1, 3 }, 4.0F);
		expect(broadcast_quotient.shape() == Shape{ 2, 3 },
			   "division broadcasts compatible input shapes");
		expect(broadcast_quotient.dtype() == DType::Float32 &&
				   broadcast_quotient.device() == Device::cpu(),
			   "division preserves the input dtype and device");

		constexpr std::array<Axis, 2> swapped_axes{ 1, 0 };
		const Tensor permuted_matrix = permute(matrix, swapped_axes);
		expect(permuted_matrix.shape() == Shape{ 3, 2 },
			   "permutation reorders tensor dimensions");
		expect(permuted_matrix.dtype() == matrix.dtype() &&
				   permuted_matrix.device() == matrix.device(),
			   "permutation preserves tensor dtype and device");

		const Tensor contiguous_matrix = contiguous(permuted_matrix);
		expect(contiguous_matrix.shape() == permuted_matrix.shape() &&
				   contiguous_matrix.dtype() == permuted_matrix.dtype() &&
				   contiguous_matrix.device() == permuted_matrix.device(),
			   "contiguous preserves tensor metadata");

		constexpr std::array<Axis, 2> negative_swapped_axes{ -1, -2 };
		expect(permute(matrix, negative_swapped_axes).shape() == Shape{ 3, 2 },
			   "permutation accepts normalized negative axes");
		expect(permute(scalar, std::span<const Axis>{}).shape().is_scalar(),
			   "an empty permutation preserves a scalar tensor");

		const Tensor volume = full(Shape{ 2, 3, 4 }, 1.0F);
		expect(transpose(volume).shape() == Shape{ 4, 3, 2 },
			   "transpose without axes reverses all dimensions");
		expect(transpose(volume, 0, -1).shape() == Shape{ 4, 3, 2 },
			   "transpose swaps two normalized axes");
		expect(transpose(matrix).shape() == Shape{ 3, 2 },
			   "transpose reverses matrix dimensions");
		expect(transpose(scalar).shape().is_scalar(),
			   "transpose preserves a scalar tensor");

		const Tensor reshaped_matrix = reshape(matrix, Shape{ 3, 2 });
		expect(reshaped_matrix.shape() == Shape{ 3, 2 },
			   "reshape replaces the tensor shape while preserving its element count");
		expect(reshaped_matrix.dtype() == matrix.dtype() &&
				   reshaped_matrix.device() == matrix.device(),
			   "reshape preserves tensor dtype and device");
		expect(reshape(scalar, Shape{ 1 }).shape() == Shape{ 1 },
			   "reshape can convert a scalar into a ranked singleton tensor");
		expect(reshape(empty, Shape{ 0, 6 }).shape() == Shape{ 0, 6 },
			   "reshape accepts a different empty shape");

		expect(flatten(matrix).shape() == Shape{ 6 },
			   "flatten without axes produces a vector");
		expect(flatten(volume, 1).shape() == Shape{ 2, 12 },
			   "flatten combines an inclusive range of dimensions");
		expect(flatten(volume, -2, -1).shape() == Shape{ 2, 12 },
			   "flatten normalizes negative axes");
		expect(flatten(volume, 1, 1).shape() == volume.shape(),
			   "flattening one axis preserves the input shape");
		expect(flatten(scalar).shape() == Shape{ 1 },
			   "flatten converts a scalar to a one-element vector");
		expect(flatten(empty).shape() == Shape{ 0 },
			   "flatten preserves an empty tensor's element count");

		const Tensor singleton_dimensions = full(Shape{ 1, 2, 1, 3 }, 1.0F);
		expect(squeeze(singleton_dimensions).shape() == Shape{ 2, 3 },
			   "squeeze without axes removes every singleton dimension");
		expect(squeeze(singleton_dimensions, 0).shape() == Shape{ 2, 1, 3 },
			   "squeeze removes one selected singleton dimension");
		constexpr std::array<Axis, 2> squeezed_axes{ 0, -2 };
		expect(squeeze(singleton_dimensions, squeezed_axes).shape() == Shape{ 2, 3 },
			   "squeeze accepts multiple normalized singleton axes");
		expect(squeeze(scalar).shape().is_scalar(),
			   "squeeze preserves a scalar tensor");

		expect(unsqueeze(matrix, 0).shape() == Shape{ 1, 2, 3 },
			   "unsqueeze inserts a leading singleton dimension");
		expect(unsqueeze(matrix, -1).shape() == Shape{ 2, 3, 1 },
			   "unsqueeze accepts a negative trailing insertion axis");
		expect(unsqueeze(scalar, 0).shape() == Shape{ 1 },
			   "unsqueeze converts a scalar to a singleton vector");

		const Tensor broadcast_source = full(Shape{ 2, 1 }, 3.0F);
		const Tensor broadcasted_tensor = broadcast_to(broadcast_source, Shape{ 2, 3 });
		expect(broadcasted_tensor.shape() == Shape{ 2, 3 },
			   "broadcast_to expands singleton dimensions to the requested shape");
		expect(broadcasted_tensor.dtype() == broadcast_source.dtype() &&
				   broadcasted_tensor.device() == broadcast_source.device(),
			   "broadcast_to preserves tensor dtype and device");
		expect(broadcast_to(scalar, Shape{ 2, 3 }).shape() == Shape{ 2, 3 },
			   "broadcast_to expands a scalar to a ranked shape");
		expect(broadcast_to(empty, Shape{ 4, 2, 0, 3 }).shape() == Shape{ 4, 2, 0, 3 },
			   "broadcast_to supports compatible empty shapes and leading dimensions");

		const std::array<Tensor, 3> concatenation_inputs{
			full(Shape{2, 1}, 1.0F),
			full(Shape{2, 0}, 2.0F),
			full(Shape{2, 3}, 3.0F) };
		expect(concatenate(concatenation_inputs, -1).shape() == Shape{ 2, 4 },
			   "concatenate joins extents along a normalized axis and accepts empty inputs");

		constexpr std::array<Axis, 1> last_axis{ 1 };
		const Tensor row_sums = sum(matrix, last_axis);
		expect(row_sums.shape() == Shape{ 2 },
			   "sum removes a reduced axis from the output shape");
		expect(row_sums.dtype() == matrix.dtype() && row_sums.device() == matrix.device(),
			   "sum preserves tensor dtype and device");
		expect(sum(matrix, last_axis, true).shape() == Shape{ 2, 1 },
			   "sum with keep_dim retains a reduced axis as a singleton dimension");

		constexpr std::array<Axis, 2> unsorted_axes{ -1, 0 };
		expect(sum(full(Shape{ 2, 3, 4 }, 1.0F), unsorted_axes).shape() == Shape{ 3 },
			   "sum normalizes negative axes and accepts them in any order");
		expect(sum(matrix, std::span<const Axis>{}).shape() == matrix.shape(),
			   "sum over no axes preserves the input shape");
		expect(sum(matrix).shape().is_scalar(),
			   "sum without explicit axes reduces every matrix axis");
		expect(sum(matrix, true).shape() == Shape{ 1, 1 },
			   "sum with keep_dim retains every implicitly reduced axis");
		expect(sum(scalar).shape().is_scalar(),
			   "sum without explicit axes preserves a scalar shape");

		constexpr std::array<Axis, 1> empty_reduction_axis{ 1 };
		expect(sum(empty, empty_reduction_axis).shape() == Shape{ 2, 3 },
			   "sum removes a zero-length reduction axis while preserving the other axes");

		expect(matmul(full(Shape{ 3 }, 1.0F), full(Shape{ 3 }, 1.0F)).shape().is_scalar(),
			   "vector-vector matmul produces a scalar");
		expect(matmul(full(Shape{ 2, 3 }, 1.0F), full(Shape{ 3 }, 1.0F)).shape() == Shape{ 2 },
			   "matrix-vector matmul produces a vector of matrix rows");
		expect(matmul(full(Shape{ 3 }, 1.0F), full(Shape{ 3, 4 }, 1.0F)).shape() == Shape{ 4 },
			   "vector-matrix matmul produces a vector of matrix columns");
		const Tensor matrix_product =
			matmul(full(Shape{ 2, 3 }, 1.0F), full(Shape{ 3, 4 }, 1.0F));
		expect(matrix_product.shape() == Shape{ 2, 4 } &&
				   matrix_product.dtype() == DType::Float32 &&
				   matrix_product.device() == Device::cpu(),
			   "matrix-matrix matmul infers its output specification");

		expect_throws<std::invalid_argument>(
			[]
			{
				(void)matmul(full(Shape{}, 1.0F), full(Shape{ 1 }, 1.0F));
			},
			"matmul rejects a scalar input");
		expect(matmul(
			full(Shape{ 1, 1, 1 }, 1.0F),
			full(Shape{ 1 }, 1.0F))
					   .shape() == Shape{ 1, 1 },
			   "matmul infers a batched matrix-vector output specification");
		expect(matmul(
			full(Shape{ 2, 1, 3, 4 }, 1.0F),
			full(Shape{ 5, 4, 6 }, 1.0F))
					   .shape() == Shape{ 2, 5, 3, 6 },
			   "matmul broadcasts independently aligned batch dimensions");
		expect_throws<std::invalid_argument>(
			[]
			{
				(void)matmul(full(Shape{ 2, 3 }, 1.0F), full(Shape{ 2, 4 }, 1.0F));
			},
			"matmul rejects mismatched contraction dimensions");
		expect_throws<std::invalid_argument>(
			[]
			{
				(void)matmul(
					full(Shape{ 2, 3, 4 }, 1.0F),
					full(Shape{ 5, 4, 6 }, 1.0F));
			},
			"matmul rejects incompatible batch dimensions");
		expect_throws<std::invalid_argument>(
			[]
			{
				const Tensor lhs = full(
					Shape{ 2, 2 }, 1.0F,
					TensorOptions{ DType::Float32, Device::cpu(0) });
				const Tensor rhs = full(
					Shape{ 2, 2 }, 1.0F,
					TensorOptions{ DType::Float32, Device::cpu(1) });
				(void)matmul(lhs, rhs);
			},
			"matmul rejects tensors on different devices");

		expect_throws<std::invalid_argument>(
			[]
			{
				(void)(full(Shape{ 2, 3 }, 1.0F) + full(Shape{ 2, 2 }, 1.0F));
			},
			"addition rejects incompatible shapes");
		expect_throws<std::invalid_argument>(
			[]
			{
				const Tensor lhs = full(Shape{ 2 }, 1.0F, TensorOptions{ DType::Float32, Device::cpu(0) });
				const Tensor rhs = full(Shape{ 2 }, 1.0F, TensorOptions{ DType::Float32, Device::cpu(1) });
				(void)(lhs + rhs);
			},
			"addition rejects tensors on different devices");
		expect_throws<std::invalid_argument>(
			[]
			{
				(void)(full(Shape{ 2, 3 }, 1.0F) - full(Shape{ 2, 2 }, 1.0F));
			},
			"subtraction rejects incompatible shapes");
		expect_throws<std::invalid_argument>(
			[]
			{
				const Tensor lhs = full(Shape{ 2 }, 1.0F, TensorOptions{ DType::Float32, Device::cpu(0) });
				const Tensor rhs = full(Shape{ 2 }, 1.0F, TensorOptions{ DType::Float32, Device::cpu(1) });
				(void)(lhs - rhs);
			},
			"subtraction rejects tensors on different devices");
		expect_throws<std::invalid_argument>(
			[]
			{
				(void)(full(Shape{ 2, 3 }, 1.0F) * full(Shape{ 2, 2 }, 1.0F));
			},
			"multiplication rejects incompatible shapes");
		expect_throws<std::invalid_argument>(
			[]
			{
				const Tensor lhs = full(Shape{ 2 }, 1.0F, TensorOptions{ DType::Float32, Device::cpu(0) });
				const Tensor rhs = full(Shape{ 2 }, 1.0F, TensorOptions{ DType::Float32, Device::cpu(1) });
				(void)(lhs * rhs);
			},
			"multiplication rejects tensors on different devices");
		expect_throws<std::invalid_argument>(
			[]
			{
				(void)(full(Shape{ 2, 3 }, 1.0F) / full(Shape{ 2, 2 }, 1.0F));
			},
			"division rejects incompatible shapes");
		expect_throws<std::invalid_argument>(
			[]
			{
				const Tensor lhs = full(Shape{ 2 }, 1.0F, TensorOptions{ DType::Float32, Device::cpu(0) });
				const Tensor rhs = full(Shape{ 2 }, 1.0F, TensorOptions{ DType::Float32, Device::cpu(1) });
				(void)(lhs / rhs);
			},
			"division rejects tensors on different devices");
		expect_throws<std::invalid_argument>(
			[&matrix]
			{
				constexpr std::array<Axis, 1> wrong_size{ 0 };
				(void)permute(matrix, wrong_size);
			},
			"permutation rejects an axis count that differs from the input rank");
		expect_throws<std::invalid_argument>(
			[&matrix]
			{
				constexpr std::array<Axis, 2> duplicate_axes{ 0, 0 };
				(void)permute(matrix, duplicate_axes);
			},
			"permutation rejects duplicate axes");
		expect_throws<std::out_of_range>(
			[&matrix]
			{
				constexpr std::array<Axis, 2> out_of_range_axes{ 0, 2 };
				(void)permute(matrix, out_of_range_axes);
			},
			"permutation rejects an axis outside the input rank");
		expect_throws<std::out_of_range>(
			[&matrix]
			{
				(void)transpose(matrix, 0, 2);
			},
			"transpose rejects an axis outside the input rank");
		expect_throws<std::invalid_argument>(
			[&matrix]
			{
				(void)reshape(matrix, Shape{ 5 });
			},
			"reshape rejects a shape with a different element count");
		expect_throws<std::invalid_argument>(
			[&volume]
			{
				(void)flatten(volume, 2, 1);
			},
			"flatten rejects a reversed axis range");
		expect_throws<std::out_of_range>(
			[&volume]
			{
				(void)flatten(volume, 0, 3);
			},
			"flatten rejects an axis outside the input rank");
		expect_throws<std::invalid_argument>(
			[&singleton_dimensions]
			{
				(void)squeeze(singleton_dimensions, 1);
			},
			"squeeze rejects a selected dimension that is not a singleton");
		expect_throws<std::invalid_argument>(
			[&singleton_dimensions]
			{
				constexpr std::array<Axis, 2> duplicate_axes{ 0, -4 };
				(void)squeeze(singleton_dimensions, duplicate_axes);
			},
			"squeeze rejects axes that become duplicates after normalization");
		expect_throws<std::out_of_range>(
			[&matrix]
			{
				(void)unsqueeze(matrix, 3);
			},
			"unsqueeze rejects an insertion axis outside the output rank");
		expect_throws<std::invalid_argument>(
			[&matrix]
			{
				(void)broadcast_to(matrix, Shape{ 2, 2 });
			},
			"broadcast_to rejects incompatible dimensions");
		expect_throws<std::invalid_argument>(
			[&matrix]
			{
				(void)broadcast_to(matrix, Shape{ 3 });
			},
			"broadcast_to rejects a target with fewer dimensions");
		expect_throws<std::invalid_argument>(
			[]
			{
				(void)concatenate(std::span<const Tensor>{});
			},
			"concatenate rejects an empty input sequence");
		expect_throws<std::out_of_range>(
			[&scalar]
			{
				const std::array<Tensor, 2> inputs{ scalar, scalar };
				(void)concatenate(inputs);
			},
			"concatenate rejects scalar tensors because they have no axis");
		expect_throws<std::invalid_argument>(
			[&matrix]
			{
				const std::array<Tensor, 2> inputs{
					matrix, full(Shape{3, 3}, 1.0F) };
				(void)concatenate(inputs, 1);
			},
			"concatenate rejects mismatched non-concatenated extents");
		expect_throws<std::invalid_argument>(
			[&matrix]
			{
				constexpr std::array<Axis, 2> duplicate_axes{ 1, -1 };
				(void)sum(matrix, duplicate_axes);
			},
			"sum rejects axes that become duplicates after normalization");
		expect_throws<std::out_of_range>(
			[&matrix]
			{
				constexpr std::array<Axis, 1> out_of_range_axis{ 2 };
				(void)sum(matrix, out_of_range_axis);
			},
			"sum rejects an axis outside the input rank");
		expect_throws<std::invalid_argument>(
			[&matrix]
			{
				constexpr std::array<Axis, 3> too_many_axes{ 0, 1, 0 };
				(void)sum(matrix, too_many_axes);
			},
			"sum rejects more reduction axes than the input rank");
	}
}
