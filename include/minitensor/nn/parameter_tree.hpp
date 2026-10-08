#pragma once

#include <concepts>
#include <functional>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <minitensor/nn/parameter.hpp>
#include <minitensor/tensor.hpp>

namespace minitensor::nn
{
	namespace detail
	{
		template <typename Function>
		class ConstParameterTraversal;

		template <typename Function>
		class ParameterTransformTraversal;
	}

	// Access point for a parameter tree's private structural hook. Trees participate
	// by friending this class and defining a static visit_members(Self&, Visitor&)
	class ParameterTreeAccess final
	{
	private:
		template <typename Tree, typename Visitor>
		static void visit(Tree& tree, Visitor& visitor)
		{
			using TreeType = std::remove_cvref_t<Tree>;
			TreeType::visit_members(tree, visitor);
		}

		template <typename Function>
		friend class detail::ConstParameterTraversal;

		template <typename Function>
		friend class detail::ParameterTransformTraversal;
	};

	// Specific traversal methods. Each uses ParameterTreeAccess to enter a tree.
	// Each must implement the visit, parameter, and child functions
	namespace detail
	{
		template <typename Function>
		class ConstParameterTraversal final
		{
		public:
			explicit ConstParameterTraversal(Function& function) noexcept
				: function_{ function }
			{}

			template <typename Tree>
			void visit(const Tree& tree)
			{
				ParameterTreeAccess::visit(tree, *this);
			}

			void parameter(std::string_view, const Parameter& parameter)
			{
				std::invoke(function_, parameter);
			}

			template <typename Tree>
			void child(std::string_view, const Tree& tree)
			{
				visit(tree);
			}

		private:
			Function& function_;
		};

		template <typename Function>
		class ParameterTransformTraversal final
		{
		public:
			explicit ParameterTransformTraversal(Function& function) noexcept
				: function_{ function }
			{}

			template <typename Tree>
			void visit(Tree& tree)
			{
				ParameterTreeAccess::visit(tree, *this);
			}

			void parameter(std::string_view, Parameter& parameter)
			{
				auto position = transformed_values_.find(parameter.id());
				if (position == transformed_values_.end())
				{
					Tensor transformed_value =
						std::invoke(function_, std::as_const(parameter));

					// Validate that the transformed value is valid for the parameter
					parameter = parameter.with_value(transformed_value);
					transformed_values_.emplace(
						parameter.id(),
						std::move(transformed_value));
				}
				else
				{
					// Assign the same Tensor handle if the parameter is already seen
					parameter = parameter.with_value(position->second);
				}
			}

			template <typename Tree>
			void child(std::string_view, Tree& tree)
			{
				visit(tree);
			}

		private:
			Function& function_;
			std::unordered_map<ParameterId, Tensor> transformed_values_;
		};
	}

	template <typename Function>
	concept ParameterCallback = std::invocable<Function&, const Parameter&>;

	template <typename Function>
	concept ParameterValueTransform =
		ParameterCallback<Function> && std::convertible_to<
		std::invoke_result_t<Function&, const Parameter&>,
		Tensor>;

	// Visits every parameter in depth-first hook order.
	template <typename Tree, ParameterCallback Function>
	void for_each_parameter(const Tree& tree, Function&& function)
	{
		using FunctionType = std::remove_reference_t<Function>;
		detail::ConstParameterTraversal<FunctionType> traversal{ function };
		traversal.visit(tree);
	}

	// Visits every unique parameter in depth-first hook order
	template <typename Tree, ParameterCallback Function>
	void for_each_unique_parameter(const Tree& tree, Function&& function)
	{
		std::unordered_set<ParameterId> seen;

		for_each_parameter(tree, [&](const Parameter& parameter)
		{
			if (seen.insert(parameter.id()).second)
			{
				std::invoke(function, parameter);
			}
		});
	}

	template <typename Tree, ParameterValueTransform Function>
	Tree transform_parameter_values(Tree tree, Function&& function)
	{
		using FunctionType = std::remove_reference_t<Function>;
		detail::ParameterTransformTraversal<FunctionType> traversal{ function };
		traversal.visit(tree);
		return tree;
	}
}
