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

class ParameterMap
{
  public:
    using Map = std::unordered_map<ParameterId, ParameterState>;

    explicit ParameterMap(Map states);
    [[nodiscard]] const ParameterState &get(ParameterId id) const;
    [[nodiscard]] bool contains(ParameterId id) const;
    [[nodiscard]] std::size_t size() const noexcept;

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
};

template <typename Tree>
class ParameterSnapshot
{
  public:
    ParameterSnapshot(Tree tree, ParameterMap parameters) : tree_{std::move(tree)}, parameters_{std::move(parameters)}
    {
        std::unordered_set<ParameterId> referenced_ids;

        // Validate that the parameter tree and the flat map agree.
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

  private:
    Tree tree_;
    ParameterMap parameters_;
};

class ParameterBuilder
{
  public:
    ParameterBuilder() = default;
    ParameterBuilder(const ParameterBuilder &) = delete;
    ParameterBuilder &operator=(const ParameterBuilder &) = delete;
    ParameterBuilder(ParameterBuilder &&) noexcept = default;
    ParameterBuilder &operator=(ParameterBuilder &&) noexcept = default;

    [[nodiscard]] ParameterId add(ParameterState state);

    // Finalize once at the root, after all children have used this builder.
    template <typename Tree>
    [[nodiscard]] ParameterSnapshot<Tree> finish(Tree tree) &&
    {
        return ParameterSnapshot<Tree>{std::move(tree), ParameterMap{std::move(parameters_)}};
    }

  private:
    static ParameterId make_id();

    ParameterMap::Map parameters_;
};

// Transforms each unique state once while retaining the ID tree.
// Map iteration order is unspecified.
template <typename Tree, typename Transform>
[[nodiscard]] ParameterSnapshot<Tree> transform_parameters(ParameterSnapshot<Tree> snapshot, Transform &&transform)
{
    ParameterMap::Map transformed;

    transformed.reserve(snapshot.parameters().size());

    for (const auto &[id, state] : snapshot.parameters())
    {
        transformed.emplace(id, std::invoke(transform, id, state));
    }
    return ParameterSnapshot<Tree>{snapshot.tree(), ParameterMap{std::move(transformed)}};
}

} // namespace minitensor::nn
