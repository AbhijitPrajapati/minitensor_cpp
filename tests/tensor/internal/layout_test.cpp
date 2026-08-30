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

        expect_throws<std::overflow_error>(
            []
            {
                const Shape shape{0, std::numeric_limits<Extent>::max(), 2};
                (void)Layout::contiguous(shape);
            },
            "contiguous layout construction rejects stride overflow");
    }
}
