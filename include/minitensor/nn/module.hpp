#pragma once

#include <concepts>
#include <functional>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <unordered_map>
#include <utility>

#include <minitensor/nn/parameter.hpp>
namespace minitensor::nn
{
	namespace detail
	{
		template <typename Function>
		class ConstParameterTraversal;

		template <typename Function>
		class ParameterTransformTraversal;
	}

	// Access point for a module's private structural hook. Modules participate
	// by friending this class and defining a static visit_members(Self&, Visitor&)
	class ModuleAccess final
	{
	private:
		template <typename Module, typename Visitor>
		static void visit(Module& module, Visitor& visitor)
		{
			using ModuleType = std::remove_cvref_t<Module>; // Remove any const or references
			ModuleType::visit_members(module, visitor);
		}

		template <typename Function>
		friend class detail::ConstParameterTraversal;

		template <typename Function>
		friend class detail::ParameterTransformTraversal;
	};

	// Specific parameter traversal methods. These use ModuleAccess to visit parameters.
	namespace detail
	{
		template <typename Function>
		class ConstParameterTraversal final
		{
		public:
			explicit ConstParameterTraversal(Function& function) noexcept
				: function_{ function }
			{}

			template <typename Module>
			void visit(const Module& module)
			{
				ModuleAccess::visit(module, *this);
			}

			void parameter(std::string_view, const Parameter& parameter)
			{
				std::invoke(function_, parameter);
			}

			template <typename Module>
			void child(std::string_view, const Module& module)
			{
				visit(module);
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

			template <typename Module>
			void visit(Module& module)
			{
				ModuleAccess::visit(module, *this);
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
					position = transformed_values_.emplace(
													  parameter.id(),
													  std::move(transformed_value))
						.first;
				}
				else
				{
					// Assign the same Tensor handle if the parameter is already seen
					parameter = parameter.with_value(position->second);
				}
			}

			template <typename Module>
			void child(std::string_view, Module& module)
			{
				visit(module);
			}

		private:
			Function& function_;
			std::unordered_map<ParameterId, Tensor> transformed_values_;
		};
	}

	// Visits every parameter in depth-first hook order.
	template <typename Module, typename Function>
		requires std::invocable<Function&, const Parameter&>
	void for_each_parameter(const Module& module, Function&& function)
	{
		using FunctionType = std::remove_reference_t<Function>;
		detail::ConstParameterTraversal<FunctionType> traversal{ function };
		traversal.visit(module);
	}

	// Visits every unique parameter in depth-first hook order
	template <typename Module, typename Function>
	void for_each_unique_parameter(const Module& module, Function&& function)
	{
		std::unordered_set<ParameterId> seen;

		for_each_parameter(module, [&](const Parameter& parameter)
						   {
							   if (seen.insert(parameter.id()).second)
							   {
								   std::invoke(function, parameter);
							   }
 });
	}

	template <typename Module, typename Function>
		requires std::invocable<Function&, const Parameter&>&& std::convertible_to<std::invoke_result_t<Function&, const Parameter&>, Tensor>
	Module transform_parameter_values(Module module, Function&& function)
	{
		using FunctionType = std::remove_reference_t<Function>;
		detail::ParameterTransformTraversal<FunctionType> traversal{ function };
		traversal.visit(module);
		return module;
	}
}
