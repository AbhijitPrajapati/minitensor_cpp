#include <minitensor/types.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

#include "tensor/storage/layout.hpp"

#include "../support/test.hpp"

namespace minitensor::test
{
    void run_layout_test()
    {
        using detail::Layout;

        const Layout scalar;
        expect(scalar.rank() == 0, "a default layout has rank zero");
        expect(scalar.strides().empty(), "a default layout has no strides");
        expect(scalar.offset() == 0, "a default layout has zero offset");
        expect(scalar.is_contiguous(Shape{}), "the default scalar layout is contiguous");

        const Layout custom{{12, 4, 1}, 5};
        const std::array<Layout::stride_type, 3> expected_strides{12, 4, 1};
        expect(custom.rank() == expected_strides.size(), "layout construction preserves rank");
        expect(std::ranges::equal(custom.strides(), expected_strides), "layout construction preserves strides");
        expect(custom.stride(1) == 4, "layout stride returns the selected stride");
        expect(custom.offset() == 5, "layout construction preserves its offset");
        expect(custom == Layout({12, 4, 1}, 5), "equal layouts compare equal");
        expect(custom != Layout({12, 4, 1}, 4), "layout offsets participate in equality");

        expect_throws<std::invalid_argument>(
            []
            {
                const Layout invalid{{1}, -1};
                (void)invalid;
            },
            "layout construction rejects negative offsets");

        const Layout contiguous = Layout::contiguous(Shape{2, 3, 4});
        expect(contiguous == Layout({12, 4, 1}), "contiguous constructs row-major strides");
        expect(contiguous.is_contiguous(Shape{2, 3, 4}), "generated row-major strides are contiguous");

        const Layout singleton_strides{{3, 99, 1}, 7};
        expect(singleton_strides.is_contiguous(Shape{2, 1, 3}),
               "strides on singleton dimensions do not affect contiguity");
        expect(!Layout({1, 2}).is_contiguous(Shape{2, 3}), "non-row-major strides are not contiguous");
        expect(!Layout({3, 1}).is_contiguous(Shape{2, 1, 3}),
               "layout and shape ranks must match for contiguity");
        expect(Layout({123, -45}).is_contiguous(Shape{4, 0}),
               "any matching-rank layout of an empty tensor is contiguous");

        const Layout broadcast_source{{3, 3, 1}, 5};
        const Layout broadcasted = broadcast_source.broadcasted_to(Shape{2, 1, 3}, Shape{2, 4, 3});
        expect(broadcasted == Layout({3, 0, 1}, 5),
               "broadcasting a layout zeroes the stride of an expanded singleton dimension");

        const Layout promoted = Layout({1}, 2).broadcasted_to(Shape{3}, Shape{4, 3});
        expect(promoted == Layout({0, 1}, 2),
               "broadcasting a layout adds zero strides for missing leading dimensions");

        const Layout scalar_broadcast = Layout{}.broadcasted_to(Shape{}, Shape{2, 3});
        expect(scalar_broadcast == Layout({0, 0}), "broadcasting a scalar produces all-zero strides");

        expect_throws<std::invalid_argument>(
            []
            {
                (void)Layout({1}).broadcasted_to(Shape{2, 3}, Shape{2, 3});
            },
            "layout broadcasting rejects a source shape with a mismatched rank");
        expect_throws<std::invalid_argument>(
            []
            {
                (void)Layout({3, 1}).broadcasted_to(Shape{2, 3}, Shape{3});
            },
            "layout broadcasting rejects a lower-rank target");
        expect_throws<std::invalid_argument>(
            []
            {
                (void)Layout({3, 1}).broadcasted_to(Shape{2, 3}, Shape{2, 4});
            },
            "layout broadcasting rejects incompatible extents");

        expect_throws<std::overflow_error>(
            []
            {
                const Shape shape{0, std::numeric_limits<Extent>::max(), 2};
                (void)Layout::contiguous(shape);
            },
            "contiguous layout construction rejects stride overflow");
    }
}
