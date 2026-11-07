#include <minitensor/nn/optimizers/adamw.hpp>

#include <cstdint>
#include <stdexcept>

#include <gtest/gtest.h>

#include "persistence/support/record.hpp"

namespace minitensor::nn::test
{
using persistence::test::support::replace_field;

TEST(AdamWConfigurationPersistenceTest, RoundTripsAdamWConfiguration)
{
    const nn::AdamW optimizer{0.1F, 0.2F, 0.5F, 0.25F, 0.01F};
    const auto restored = restore_adamw_configuration(capture_optimizer_configuration(optimizer));

    EXPECT_FLOAT_EQ(restored.learning_rate(), optimizer.learning_rate());
    EXPECT_FLOAT_EQ(restored.weight_decay(), optimizer.weight_decay());
    EXPECT_FLOAT_EQ(restored.beta1(), optimizer.beta1());
    EXPECT_FLOAT_EQ(restored.beta2(), optimizer.beta2());
    EXPECT_FLOAT_EQ(restored.epsilon(), optimizer.epsilon());
}

TEST(AdamWConfigurationPersistenceTest, RejectsInvalidConfigurationRecords)
{
    const auto record = capture_optimizer_configuration(nn::AdamW{0.1F, 0.2F});

    EXPECT_THROW((void)restore_adamw_configuration(replace_field(record, "kind", "adamw_state")),
                 std::invalid_argument);
    EXPECT_THROW((void)restore_adamw_configuration(replace_field(record, "version", std::uint64_t{2})),
                 std::invalid_argument);
    EXPECT_THROW((void)restore_adamw_configuration(replace_field(record, "learning_rate", "0.1")), std::logic_error);
    EXPECT_THROW((void)restore_adamw_configuration(replace_field(record, "beta1", 1.0F)), std::invalid_argument);
}
} // namespace minitensor::nn::test
