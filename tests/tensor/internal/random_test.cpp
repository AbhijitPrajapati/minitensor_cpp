#include <minitensor/data.hpp>
#include <minitensor/ops/creation.hpp>
#include <minitensor/random.hpp>
#include <minitensor/types.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "tensor/core/random.hpp"
#include "tensor/execution/generator_registry.hpp"
#include "tensor/graph/node.hpp"
#include "tensor/graph/value.hpp"
#include "tensor/primitives/creation/normal.hpp"
#include "tensor/primitives/creation/uniform.hpp"
#include "tensor/tensor_access.hpp"

#include "../support/test.hpp"

namespace minitensor::test
{
    namespace
    {
        const detail::Primitive &producer_primitive(const Tensor &tensor)
        {
            const detail::Node *producer = detail::TensorAccess::value(tensor)->producer();
            if (producer == nullptr)
            {
                throw TestFailure{"a random tensor must have a producer"};
            }
            return producer->primitive();
        }
    }

    void run_random_test()
    {
        using detail::Generator;
        using detail::GeneratorRegistry;
        using detail::NormalPrimitive;
        using detail::RandomKey;
        using detail::UniformPrimitive;

        Generator generator{17};
        expect(generator.reserve_key() == RandomKey{17, 0},
               "a generator reserves its seed and first stream");
        expect(generator.reserve_key() == RandomKey{17, 1},
               "successive reservations use distinct streams");
        generator.manual_seed(29);
        expect(generator.reserve_key() == RandomKey{29, 0},
               "re-seeding resets the generator stream");

        GeneratorRegistry registry{41};
        expect(registry.reserve_key(Device::cpu()) == RandomKey{41, 0},
               "a generator registry lazily creates a device generator");
        expect(registry.reserve_key(Device::cpu(1)) == RandomKey{41, 0},
               "different devices have independent streams");
        expect(registry.reserve_key(Device::cpu()) == RandomKey{41, 1},
               "a registry advances the selected device stream");
        registry.manual_seed(53);
        expect(registry.reserve_key(Device::cpu()) == RandomKey{53, 0} &&
                   registry.reserve_key(Device::cpu(1)) == RandomKey{53, 0} &&
                   registry.reserve_key(Device::cpu(2)) == RandomKey{53, 0},
               "registry re-seeding resets existing and future device generators");

        manual_seed(67);
        const Tensor uniform_tensor = uniform(Shape{2, 3}, -1.0F, 2.0F);
        const Tensor normal_tensor = normal(Shape{4}, 1.5F, 0.25F);

        const auto *uniform_primitive =
            dynamic_cast<const UniformPrimitive *>(&producer_primitive(uniform_tensor));
        const auto *normal_primitive =
            dynamic_cast<const NormalPrimitive *>(&producer_primitive(normal_tensor));
        expect(uniform_primitive != nullptr && normal_primitive != nullptr,
               "random public operations retain their concrete primitives");
        expect(uniform_primitive->low() == -1.0F && uniform_primitive->high() == 2.0F &&
                   normal_primitive->mean() == 1.5F && normal_primitive->std_dev() == 0.25F,
               "random primitives retain their validated distribution parameters");
        expect(uniform_primitive->key() == RandomKey{67, 0} &&
                   normal_primitive->key() == RandomKey{67, 1},
               "random public operations reserve ordered immutable keys");

        manual_seed(67);
        const Tensor repeated = uniform(Shape{1}, 0.0F, 1.0F);
        const auto *repeated_primitive =
            dynamic_cast<const UniformPrimitive *>(&producer_primitive(repeated));
        expect(repeated_primitive != nullptr && repeated_primitive->key() == uniform_primitive->key(),
               "public re-seeding reproduces operation key reservation");

        expect_throws<std::invalid_argument>(
            []
            {
                (void)uniform(Shape{1}, 2.0F, 1.0F);
            },
            "uniform rejects reversed bounds");
        expect_throws<std::invalid_argument>(
            []
            {
                (void)uniform(
                    Shape{1},
                    0.0F,
                    std::numeric_limits<float>::infinity());
            },
            "uniform rejects non-finite bounds");
        expect_throws<std::invalid_argument>(
            []
            {
                (void)normal(Shape{1}, 0.0F, -1.0F);
            },
            "normal rejects a negative standard deviation");
        expect_throws<std::invalid_argument>(
            []
            {
                (void)normal(
                    Shape{1},
                    std::numeric_limits<float>::quiet_NaN(),
                    1.0F);
            },
            "normal rejects non-finite parameters");

        manual_seed(71);
        expect_throws<std::invalid_argument>(
            []
            {
                (void)uniform(Shape{1}, 1.0F, -1.0F);
            },
            "an invalid random operation is rejected before graph construction");
        const Tensor after_invalid = normal(Shape{1}, 0.0F, 1.0F);
        const auto *after_invalid_primitive =
            dynamic_cast<const NormalPrimitive *>(&producer_primitive(after_invalid));
        expect(after_invalid_primitive != nullptr &&
                   after_invalid_primitive->key() == RandomKey{71, 0},
               "an invalid random operation does not consume a generator key");

        const Tensor equal_bounds = uniform(Shape{1}, 3.0F, 3.0F);
        const Tensor zero_deviation = normal(Shape{1}, -2.0F, 0.0F);
        expect(item(equal_bounds) == 3.0F && item(zero_deviation) == -2.0F,
               "degenerate random distributions materialize their constant values");

        manual_seed(83);
        const Tensor ordered_uniform = uniform(Shape{9}, -3.0F, 4.0F);
        const Tensor ordered_normal = normal(Shape{9}, 2.0F, 0.5F);
        const auto normal_evaluated_first = to_vector(ordered_normal);
        const auto uniform_evaluated_second = to_vector(ordered_uniform);

        manual_seed(83);
        const Tensor repeated_uniform = uniform(Shape{9}, -3.0F, 4.0F);
        const Tensor repeated_normal = normal(Shape{9}, 2.0F, 0.5F);
        const auto uniform_evaluated_first = to_vector(repeated_uniform);
        const auto normal_evaluated_second = to_vector(repeated_normal);

        expect(uniform_evaluated_second == uniform_evaluated_first &&
                   normal_evaluated_first == normal_evaluated_second,
               "seeded random values depend on graph construction order, not evaluation order");
        expect(std::all_of(
                   uniform_evaluated_first.begin(),
                   uniform_evaluated_first.end(),
                   [](float value)
                   {
                       return value >= -3.0F && value < 4.0F;
                   }),
               "public uniform evaluation preserves its half-open bounds");
        expect(to_vector(uniform(Shape{0}, 0.0F, 1.0F)).empty(),
               "public random evaluation supports empty tensors");
    }
}
