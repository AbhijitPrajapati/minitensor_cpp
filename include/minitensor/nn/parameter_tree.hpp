#pragma once

#include <concepts>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include <minitensor/nn/parameter_id.hpp>

namespace minitensor::nn
{
namespace detail
{
template <typename Function>
class ConstParameterTraversal;

template <typename Function>
class NamedConstParameterTraversal;

} // namespace detail

// This is the access point for a parameter tree's private structural hook. Trees
// participate by friending this class and defining a static visit_members(Self&, Visitor&).
class ParameterTreeAccess final
{
  private:
    template <typename Tree, typename Visitor>
    static void visit(Tree &tree, Visitor &visitor)
    {
        using TreeType = std::remove_cvref_t<Tree>;
        TreeType::visit_members(tree, visitor);
    }

    template <typename Function>
    friend class detail::ConstParameterTraversal;

    template <typename Function>
    friend class detail::NamedConstParameterTraversal;
};

// These are specific traversal methods. Each uses ParameterTreeAccess to enter a tree.
// Each must implement the visit, parameter, and child functions.
namespace detail
{
template <typename Function>
class ConstParameterTraversal final
{
  public:
    explicit ConstParameterTraversal(Function &function) noexcept : function_{function}
    {
    }

    template <typename Tree>
    void visit(const Tree &tree)
    {
        ParameterTreeAccess::visit(tree, *this);
    }

    void parameter(std::string_view, const ParameterId parameter_id)
    {
        std::invoke(function_, parameter_id);
    }

    template <typename Tree>
    void child(std::string_view, const Tree &tree)
    {
        visit(tree);
    }

  private:
    Function &function_;
};

template <typename Function>
class NamedConstParameterTraversal final
{
  public:
    explicit NamedConstParameterTraversal(Function &function) noexcept : function_{function}
    {
    }

    template <typename Tree>
    void visit(const Tree &tree)
    {
        ParameterTreeAccess::visit(tree, *this);
    }

    void parameter(std::string_view name, const ParameterId parameter_id)
    {
        with_component(name, [&] {
            if (!visited_paths_.insert(path_).second)
            {
                throw std::logic_error{"duplicate parameter path"};
            }

            std::invoke(function_, std::string_view{path_}, parameter_id);
        });
    }

    template <typename Tree>
    void child(std::string_view name, const Tree &tree)
    {
        with_component(name, [&] { visit(tree); });
    }

  private:
    // This function validates and appends a new component to the current path
    // before invoking the callback.
    template <typename Action>
    void with_component(std::string_view name, Action &&action)
    {
        // Validate component
        if (name.empty())
        {
            throw std::logic_error{"parameter path components cannot be empty"};
        }

        if (name.find('.') != std::string_view::npos)
        {
            throw std::logic_error{"parameter path components cannot contain '.'"};
        }

        const std::size_t previous_size = path_.size();

        // The separator should only be used between two components.
        if (!path_.empty())
        {
            path_.push_back('.');
        }

        path_.append(name);

        try
        {
            std::invoke(std::forward<Action>(action));
        }

        // The path must always be resized, even in the event of an exception,
        // in order to ensure that subsequent traversals are not compromised.
        catch (...)
        {
            path_.resize(previous_size);
            throw;
        }

        path_.resize(previous_size);
    }

    Function &function_;

    // This is a buffer string used to store the path currently being constructed.
    std::string path_;

    // This is used to ensure that no complete path is visited more than once.
    std::unordered_set<std::string> visited_paths_;
};

} // namespace detail

template <typename Function>
concept ParameterCallback = std::invocable<Function &, const ParameterId>;

template <typename Function>
concept NamedParameterCallback = std::invocable<Function &, std::string_view, const ParameterId>;

// Visits every parameter in depth-first hook order
template <typename Tree, ParameterCallback Function>
void for_each_parameter(const Tree &tree, Function &&function)
{
    using FunctionType = std::remove_reference_t<Function>;
    detail::ConstParameterTraversal<FunctionType> traversal{function};
    traversal.visit(tree);
}

// Visits every unique parameter, delegating to for_each_parameter
template <typename Tree, ParameterCallback Function>
void for_each_unique_parameter(const Tree &tree, Function &&function)
{
    std::unordered_set<ParameterId> seen;

    for_each_parameter(tree, [&](const ParameterId parameter_id) {
        if (seen.insert(parameter_id).second)
        {
            std::invoke(function, parameter_id);
        }
    });
}

// Visits every parameter with a named callback constructed using its
// location in the parameter tree. This is used for persistence and
// diagnostics. A module provides a component for each of its parameters.
// A path component must not contain "." as that is the canonical path separator.
// Duplicate complete paths are invalid as well.
template <typename Tree, NamedParameterCallback Function>
void for_each_named_parameter(const Tree &tree, Function &&function)
{
    using FunctionType = std::remove_reference_t<Function>;
    detail::NamedConstParameterTraversal<FunctionType> traversal{function};
    traversal.visit(tree);
}

// Performs the named traversal above, but only for unique parameters.
template <typename Tree, NamedParameterCallback Function>
void for_each_unique_named_parameter(const Tree &tree, Function &&function)
{
    std::unordered_set<ParameterId> seen;

    for_each_named_parameter(tree, [&](std::string_view path, const ParameterId parameter_id) {
        if (seen.insert(parameter_id).second)
        {
            std::invoke(function, path, parameter_id);
        }
    });
}

} // namespace minitensor::nn
