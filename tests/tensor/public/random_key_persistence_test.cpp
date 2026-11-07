#include <minitensor/random.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include <minitensor/persistence/record.hpp>

#include "persistence/support/record.hpp"

namespace minitensor::test
{
using persistence::test::support::replace_field;

TEST(RandomKeyPersistenceTest, RoundTripsSeededAndDerivedKeys)
{
    const RandomKey seed{std::numeric_limits<std::uint64_t>::max()};
    const auto children = split(seed);
    for (const auto &key : {seed, children[0], children[1], fold_in(children[1], 42)})
    {
        auto record = capture_random_key(key);
        EXPECT_TRUE(record.contains({"kind", "version", "first", "second"}, true));
        const auto restored = restore_random_key(record);
        EXPECT_EQ(restored, key);
        EXPECT_EQ(split(restored), split(key));
    }
}

TEST(RandomKeyPersistenceTest, RejectsInvalidRecordFields)
{
    const auto record = capture_random_key(RandomKey{123});
    auto unsupported_version = replace_field(record, "version", std::uint64_t{2});
    EXPECT_THROW((void)restore_random_key(unsupported_version), std::invalid_argument);

    auto extra = record;
    extra.add("extra", true);
    EXPECT_THROW((void)restore_random_key(extra), std::invalid_argument);

    auto signed_word = replace_field(record, "second", -1);
    EXPECT_THROW((void)restore_random_key(signed_word), std::logic_error);
}
} // namespace minitensor::test
