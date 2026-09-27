#pragma once

#include <initializer_list>
#include <span>

#include <minitensor/tensor.hpp>

namespace minitensor
{
	void eval(const Tensor& tensor);
	void eval(std::span<const Tensor> tensors);
	void eval(std::initializer_list<Tensor> tensors);
}