#pragma once

#include <istream>
#include <ostream>

#include <minitensor/persistence/record.hpp>

namespace minitensor::persistence::detail
{
// Version 2 wire format. All integer fields are fixed-width little-endian.
// Header: "MTARCH\r\n" (8 bytes), version u32, reserved flags u32 (zero),
// logical payload length u64, CRC-32C of the preceding 24 bytes u32.
// Payload: consecutive frames of min(65536, remaining payload) raw bytes,
// each followed by its CRC-32C u32. No unchecked frame length is allocated.
// Footer: "MEND" (4 bytes), CRC-32C(header including its CRC + logical payload
// + "MEND") u32. Frames are verified before their bytes reach the decoder.
// The logical payload is one root record. Record/list counts and string lengths
// are u32. Each record entry is name length + name bytes + tagged value.
// Tensor metadata is dtype u8, rank u32, extents u64, then logical-order data.
// Value tags are fixed in binary_format.hpp. Bool payloads are u8 (0 or 1),
// integers and Float64 are 8 bytes, Float32 is 4 bytes. Strings are length +
// bytes, records are count + entries, and lists are count + tagged values.
// Scalar floats use IEEE-754 bits, signed integers use two's-complement bits.
// Names and strings are arbitrary bytes. Duplicate names are invalid.
// Limits and conservative allocation accounting are shared by reads and writes
// in binary_format.hpp. ReadOptions only chooses the destination tensor device.
// The writer stages the complete payload, spilling beyond
// 1 MiB to an automatically removed temporary file, before touching output.
// Destination I/O failures may still leave partial output. The caller owns
// flushing, closing, and atomic file replacement. Streams must use binary mode.
// Record::read consumes exactly one framed archive, permitting concatenation.
// Internal whole-file consumers may require EOF with ArchiveBoundary::EndOfStream.
enum class ArchiveBoundary
{
    Single,
    EndOfStream
};

void write_record(std::ostream &output, const Record &record);
[[nodiscard]] Record read_record(std::istream &input, const Record::ReadOptions &options,
                                 ArchiveBoundary boundary = ArchiveBoundary::Single);
} // namespace minitensor::persistence::detail
