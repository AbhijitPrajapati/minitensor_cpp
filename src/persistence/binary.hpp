#pragma once

#include <istream>
#include <ostream>

#include <minitensor/persistence/record.hpp>

namespace minitensor::persistence::detail
{
void write_record(std::ostream &output, const Record &record);
[[nodiscard]] Record read_record(std::istream &input, const Record::ReadOptions &options);
} // namespace minitensor::persistence::detail