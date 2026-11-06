#pragma once

#include <string_view>

#include <minitensor/persistence/record.hpp>

namespace minitensor::persistence::test::support
{
// Make a malformed record by replacing a single field in a valid record.
inline Record replace_field(const Record &record, std::string_view name, const Value &replacement)
{
    Record result;
    for (const auto &[field, value] : record)
    {
        result.add(field, field == name ? replacement : value);
    }
    return result;
}
} // namespace minitensor::persistence::test::support
