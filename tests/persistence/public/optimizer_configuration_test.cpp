#include <minitensor/persistence/optimizer_configuration.hpp>

#include <cstdint>
#include <stdexcept>

#include <gtest/gtest.h>

#include <minitensor/nn/optimizers/adamw.hpp>

#include "persistence/support/record.hpp"

namespace minitensor::persistence::test
{
using support::replace_field;

TEST(OptimizerConfigurationTest, RoundTripsAdamWConfiguration)
{
    const nn::AdamW optimizer{0.1F, 0.2F, 0.5F, 0.25F, 0.01F};
    const auto restored = restore_optimizer_config<nn::AdamW>(capture_optimizer_config(optimizer));

    EXPECT_FLOAT_EQ(restored.learning_rate(), optimizer.learning_rate());
    EXPECT_FLOAT_EQ(restored.weight_decay(), optimizer.weight_decay());
    EXPECT_FLOAT_EQ(restored.beta1(), optimizer.beta1());
    EXPECT_FLOAT_EQ(restored.beta2(), optimizer.beta2());
    EXPECT_FLOAT_EQ(restored.epsilon(), optimizer.epsilon());
}

TEST(OptimizerConfigurationTest, RejectsInvalidConfigurationRecords)
{
    const auto record = capture_optimizer_config(nn::AdamW{0.1F, 0.2F});

    EXPECT_THROW((void)restore_optimizer_config<nn::AdamW>(replace_field(record, "kind", "adamw_state")),
                 std::invalid_argument);
    EXPECT_THROW((void)restore_optimizer_config<nn::AdamW>(replace_field(record, "version", std::uint64_t{2})),
                 std::invalid_argument);
    EXPECT_THROW((void)restore_optimizer_config<nn::AdamW>(replace_field(record, "learning_rate", "0.1")),
                 std::logic_error);
    EXPECT_THROW((void)restore_optimizer_config<nn::AdamW>(replace_field(record, "beta1", 1.0F)),
                 std::invalid_argument);
}
} // namespace minitensor::persistence::test
