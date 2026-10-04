#pragma once

#include <concepts>
#include <functional>
#include <string_view>
#include <type_traits>

#include <minitensor/nn/parameter.hpp>
#include <unordered_map>
#include <vector>


namespace minitensor::nn
{
	namespace detail
	{
		template <typename Function>
		class ConstParameterTraversal;
	}

	// Access point for a module's private structural hook. Modules participate
	// by friending this class and defining a static visit_members(Self&, Visitor&)
	class ModuleAccess final
	{
	private:
		template <typename Module, typename Visitor>
		static void visit(const Module& module, Visitor& visitor)
		{
			using ModuleType = std::remove_cvref_t<Module>; // Remove any const or references
			ModuleType::visit_members(module, visitor);
		}

		template <typename Function>
		friend class detail::ConstParameterTraversal;
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

	template <typename Module, typename Function>
	void for_each_unique_parameter(const Module& module, Function&& function)
	{
		std::unordered_map<ParameterId, Parameter*> seen;
		std::vector<const Parameter*> ordered;

		for_each_parameter(module, [&](const Parameter& parameter)
 {
	 auto [position, inserted] = seen.try_emplace(parameter.id(), &parameter);
	 if (inserted)
	 {
		 ordered.push_back(&parameter);
	 }
	 else
	 {

	 }
		});

		for (const Parameter* parameter : ordered)
		{
			std::invoke(function, *parameter);
		}
	}
}
