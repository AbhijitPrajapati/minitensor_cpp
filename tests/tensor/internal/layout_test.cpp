#include <minitensor/types.hpp>

#include "tensor/storage/layout.hpp"

namespace minitensor::test
{
    using detail::Layout;

    void run_layout_test()
    {
        Layout{}.rank() == 0;
        Layout{{2, 1}}.offset() == 0;
        Layout{{2, 1}}.rank() == 2;
        Layout{{2, 1}}.stride(0) == 2;
        Layout{{2, 1}, 5}.offset() == 5;
        Layout{{2, 1}, -5}; // shouldnt work
        Layout{{3, 1}, 2} != Layout{{2, 1}, 2};
        Layout{{2, 1}, 5} != Layout{{2, 1}, 2};
        Layout{{2, 1}, 2} == Layout{{2, 1}, 2};
        Layout::contiguous(Shape{2, 2}).stride(0) == 2;
        Layout::contiguous(Shape{2, 2}).stride(1) == 1;
        
    }
}
