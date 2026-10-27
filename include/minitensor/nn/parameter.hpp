#pragma once

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <minitensor/nn/parameter_tree.hpp>
#include <minitensor/tensor.hpp>

namespace minitensor::nn
{
struct ParameterMetadata
{
    bool trainable{true};
    friend bool operator==(const ParameterMetadata &, const ParameterMetadata &) = default;
};

// The state of a parameter at any given step is independent of its ID.
struct ParameterState
{
    Tensor value;
    ParameterMetadata metadata{};
};

template <typename Tree>
class ParameterSnapshot;

// This class represents a mapping from a parameter's ID to its state
// for a particular snapshot.
class ParameterMap
{
  public:
    using Map = std::unordered_map<ParameterId, ParameterState>;

    explicit ParameterMap(ParameterMap::Map states) : states_(std::move(states))
    {
    }

    [[nodiscard]] bool contains(ParameterId id) const
    {
        return states_.contains(id);
    }

    [[nodiscard]] const ParameterState &get(ParameterId id) const
    {
        return states_.at(id);
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return states_.size();
    }

    [[nodiscard]] Map::const_iterator begin() const noexcept
    {
        return states_.begin();
    }

    [[nodiscard]] Map::const_iterator end() const noexcept
    {
        return states_.end();
    }

  private:
    Map states_;

    template <typename Tree>
    friend class ParameterSnapshot;
};

// This is the public interface representing a single state of a module's parameters,
// made up of a tree and a parameter map.
template <typename Tree>
class ParameterSnapshot
{
  public:
    ParameterSnapshot(Tree tree, ParameterMap parameters) : tree_{std::move(tree)}, parameters_{std::move(parameters)}
    {
        // The tree and the flat map must reference the exact same IDs.

        std::unordered_set<ParameterId> referenced_ids;
        for_each_parameter(tree_, [&](ParameterId id) {
            if (!parameters_.contains(id))
            {
                throw std::invalid_argument{"parameter tree references an unknown parameter ID"};
            }
            referenced_ids.insert(id);
        });

        if (referenced_ids.size() != parameters_.size())
        {
            throw std::invalid_argument{"parameter map contains unreferenced states"};
        }
    }

    [[nodiscard]] const Tree &tree() const noexcept
    {
        return tree_;
    }

    [[nodiscard]] const ParameterMap &parameters() const noexcept
    {
        return parameters_;
    }

    [[nodiscard]] const ParameterState &get(ParameterId id) const
    {
        return parameters_.get(id);
    }

    [[nodiscard]] const Tensor &value(ParameterId id) const
    {
        return get(id).value;
    }

    // Transforms each unique state once while retaining the ID tree. Moving
    // the snapshot saves allocation.
    template <typename Transform>
    [[nodiscard]] ParameterSnapshot transform(Transform &&transform) &&
    {
        for (auto &[id, state] : parameters_.states_)
        {
            state = std::invoke(transform, id, std::as_const(state));
        }
        return std::move(*this);
    }

    // Delegates to the function above with a copy.
    template <typename Transform>
    [[nodiscard]] ParameterSnapshot transform(Transform &&transform) const &
    {
        ParameterSnapshot<Tree> copy = *this;
        return std::move(copy).transform(transform);
    }

  private:
    Tree tree_;
    ParameterMap parameters_;
};

// This class facilitates the construction of an initial snapshot. An instance is passed down
// through the module tree as each parameter is registered.
class ParameterBuilder
{
  public:
    ParameterBuilder() = default;
    ParameterBuilder(const ParameterBuilder &) = delete;
    ParameterBuilder &operator=(const ParameterBuilder &) = delete;
    ParameterBuilder(ParameterBuilder &&) noexcept = default;
    ParameterBuilder &operator=(ParameterBuilder &&) noexcept = default;

    [[nodiscard]] ParameterId add(ParameterState state);

    // This function is called once at the root to finalize the parameter map and create
    // a validated snapshot.
    template <typename Tree>
    [[nodiscard]] ParameterSnapshot<Tree> finish(Tree tree) &&
    {
        return ParameterSnapshot<Tree>{std::move(tree), ParameterMap{std::move(parameters_)}};
    }

  private:
    static ParameterId make_id();

    ParameterMap::Map parameters_;
};
} // namespace minitensor::nn
