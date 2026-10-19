#include <array>
#include <unordered_set>

#include <gtest/gtest.h>

#include "tensor/graph/primitive_id.hpp"
#include "tensor/primitives/creation/full.hpp"
#include "tensor/primitives/creation/normal.hpp"
#include "tensor/primitives/creation/uniform.hpp"
#include "tensor/primitives/elementwise/absolute_value.hpp"
#include "tensor/primitives/elementwise/add.hpp"
#include "tensor/primitives/elementwise/divide.hpp"
#include "tensor/primitives/elementwise/exponential.hpp"
#include "tensor/primitives/elementwise/hyperbolic_tangent.hpp"
#include "tensor/primitives/elementwise/logaddexp.hpp"
#include "tensor/primitives/elementwise/logarithm.hpp"
#include "tensor/primitives/elementwise/multiply.hpp"
#include "tensor/primitives/elementwise/negate.hpp"
#include "tensor/primitives/elementwise/sigmoid.hpp"
#include "tensor/primitives/elementwise/sign.hpp"
#include "tensor/primitives/elementwise/square_root.hpp"
#include "tensor/primitives/elementwise/subtract.hpp"
#include "tensor/primitives/linalg/matmul.hpp"
#include "tensor/primitives/manipulation/basic_index.hpp"
#include "tensor/primitives/manipulation/broadcast_to.hpp"
#include "tensor/primitives/manipulation/concatenate.hpp"
#include "tensor/primitives/manipulation/contiguous.hpp"
#include "tensor/primitives/manipulation/gather.hpp"
#include "tensor/primitives/manipulation/permute.hpp"
#include "tensor/primitives/manipulation/reshape.hpp"
#include "tensor/primitives/reduction/logsumexp.hpp"
#include "tensor/primitives/reduction/sum.hpp"

namespace minitensor::test
{
TEST(PrimitiveIdTest, BuiltInPrimitivesHaveUniqueNonzeroIds)
{
    const std::array<detail::PrimitiveId, 28> ids{detail::FullPrimitive::ID,
                                                  detail::UniformPrimitive::ID,
                                                  detail::NormalPrimitive::ID,
                                                  detail::AbsoluteValuePrimitive::ID,
                                                  detail::SignPrimitive::ID,
                                                  detail::NegatePrimitive::ID,
                                                  detail::ExponentialPrimitive::ID,
                                                  detail::LogarithmPrimitive::ID,
                                                  detail::SquareRootPrimitive::ID,
                                                  detail::HyperbolicTangentPrimitive::ID,
                                                  detail::SigmoidPrimitive::ID,
                                                  detail::AddPrimitive::ID,
                                                  detail::SubtractPrimitive::ID,
                                                  detail::MultiplyPrimitive::ID,
                                                  detail::DividePrimitive::ID,
                                                  detail::LogAddExpPrimitive::ID,
                                                  detail::ContiguousPrimitive::ID,
                                                  detail::ReshapePrimitive::ID,
                                                  detail::PermutePrimitive::ID,
                                                  detail::BroadcastToPrimitive::ID,
                                                  detail::ConcatenatePrimitive::ID,
                                                  detail::BasicIndexPrimitive::ID,
                                                  detail::BasicIndexScatterPrimitive::ID,
                                                  detail::GatherPrimitive::ID,
                                                  detail::GatherScatterAddPrimitive::ID,
                                                  detail::SumPrimitive::ID,
                                                  detail::LogSumExpPrimitive::ID,
                                                  detail::MatmulPrimitive::ID};

    const std::unordered_set<detail::PrimitiveId, detail::PrimitiveIdHash> unique_ids{ids.begin(), ids.end()};

    EXPECT_EQ(unique_ids.size(), ids.size());
    for (const detail::PrimitiveId id : ids)
    {
        EXPECT_NE(id.value(), 0);
    }
}
} // namespace minitensor::test
