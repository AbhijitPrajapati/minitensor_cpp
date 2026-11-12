#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ios>
#include <istream>
#include <memory>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "binary.hpp"
#include "binary_format.hpp"

namespace minitensor::persistence::detail::binary
{
inline void write_exact(std::ostream &output, std::span<const std::byte> bytes)
{
    if (!std::in_range<std::streamsize>(bytes.size()))
    {
        throw std::length_error{"archive I/O transfer exceeds streamsize"};
    }
    output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output)
    {
        throw std::ios_base::failure{"failed to write archive bytes"};
    }
}

// Staging prevents validation, evaluation, or allocation failures from corrupting
// an existing destination. Memory is bounded independently of archive length.
class Spool
{
  public:
    void append(std::span<const std::byte> bytes)
    {
        if (bytes.empty())
        {
            return;
        }
        if (bytes.size() > max_archive_bytes - size_)
        {
            throw std::length_error{"archive exceeds the encoded byte budget"};
        }
        validate_archive_size(size_ + bytes.size());
        if (!file_ && bytes.size() <= spool_memory_bytes - memory_.size())
        {
            const auto required = memory_.size() + bytes.size();
            if (required > memory_.capacity())
            {
                memory_.reserve(std::min(spool_memory_bytes, std::max(required, 2 * memory_.capacity())));
            }
            memory_.insert(memory_.end(), bytes.begin(), bytes.end());
        }
        else
        {
            if (!file_)
            {
                file_.reset(std::tmpfile());
                if (!file_)
                {
                    throw std::ios_base::failure{"could not create temporary archive spool"};
                }
                write_file(memory_);
                std::vector<std::byte>{}.swap(memory_);
            }
            write_file(bytes);
        }
        size_ += bytes.size();
    }

    [[nodiscard]] std::uint64_t size() const noexcept
    {
        return size_;
    }

    void rewind()
    {
        if (file_ && (std::fflush(file_.get()) != 0 || std::fseek(file_.get(), 0, SEEK_SET) != 0))
        {
            throw std::ios_base::failure{"failed to rewind archive spool"};
        }
        position_ = 0;
    }

    void read(std::span<std::byte> bytes)
    {
        if (bytes.size() > size_ - position_)
        {
            throw std::logic_error{"archive spool read exceeds payload"};
        }
        if (file_)
        {
            if (std::fread(bytes.data(), 1, bytes.size(), file_.get()) != bytes.size())
            {
                throw std::ios_base::failure{"failed to read archive spool"};
            }
        }
        else
        {
            std::copy_n(memory_.data() + static_cast<std::size_t>(position_), bytes.size(), bytes.data());
        }
        position_ += bytes.size();
    }

  private:
    void write_file(std::span<const std::byte> bytes)
    {
        if (std::fwrite(bytes.data(), 1, bytes.size(), file_.get()) != bytes.size())
        {
            throw std::ios_base::failure{"failed to write archive spool"};
        }
    }

    struct FileCloser
    {
        void operator()(std::FILE *file) const noexcept
        {
            std::fclose(file);
        }
    };

    std::vector<std::byte> memory_;
    std::unique_ptr<std::FILE, FileCloser> file_;
    std::uint64_t size_{0};
    std::uint64_t position_{0};
};

inline void emit_archive(std::ostream &output, Spool &payload)
{
    validate_archive_size(payload.size());
    payload.rewind();
    std::array<std::byte, header_bytes> header{};
    std::copy(magic.begin(), magic.end(), header.begin());
    store_le<std::uint32_t>(std::span{header}.subspan(8, 4), version);
    store_le<std::uint64_t>(std::span{header}.subspan(16, 8), payload.size());
    store_le<std::uint32_t>(std::span{header}.subspan(24, 4), crc32c(std::span{header}.first(24)));
    Checksum checksum;
    checksum.update(header);
    write_exact(output, header);

    std::array<std::byte, frame_bytes> buffer;
    std::array<std::byte, 4> encoded_crc;
    auto remaining = payload.size();
    while (remaining != 0)
    {
        const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(remaining, buffer.size()));
        const auto frame = std::span{buffer}.first(count);
        payload.read(frame);
        checksum.update(frame);
        store_le<std::uint32_t>(encoded_crc, crc32c(frame));
        write_exact(output, frame);
        write_exact(output, encoded_crc);
        remaining -= count;
    }
    checksum.update(end_marker);
    store_le<std::uint32_t>(encoded_crc, checksum.value());
    write_exact(output, end_marker);
    write_exact(output, encoded_crc);
}

class ArchiveInput
{
  public:
    explicit ArchiveInput(std::istream &input) : input_{input}
    {
        std::array<std::byte, header_bytes> header;
        read_exact(header);
        if (!std::equal(magic.begin(), magic.end(), header.begin()))
        {
            fail("invalid archive magic");
        }
        if (crc32c(std::span{header}.first(24)) != load_le<std::uint32_t>(std::span{header}.subspan(24, 4)))
        {
            fail("archive header checksum mismatch");
        }
        if (load_le<std::uint32_t>(std::span{header}.subspan(8, 4)) != version)
        {
            fail("unsupported archive version");
        }
        if (load_le<std::uint32_t>(std::span{header}.subspan(12, 4)) != 0)
        {
            fail("unsupported archive flags");
        }
        remaining_ = load_le<std::uint64_t>(std::span{header}.subspan(16, 8));
        validate_archive_size(remaining_);
        if (remaining_ < 4)
        {
            fail("archive payload has no root record count");
        }
        checksum_.update(header);
    }

    void require(std::uint64_t bytes) const
    {
        if (bytes > remaining_)
        {
            fail("value exceeds the framed payload length");
        }
    }

    void read(std::span<std::byte> destination)
    {
        require(destination.size());
        while (!destination.empty())
        {
            if (position_ == available_)
            {
                load_frame();
            }
            const auto count = std::min(destination.size(), available_ - position_);
            std::copy_n(buffer_.data() + position_, count, destination.data());
            destination = destination.subspan(count);
            position_ += count;
            remaining_ -= count;
            consumed_ += count;
        }
    }

    [[nodiscard]] std::uint64_t remaining() const noexcept
    {
        return remaining_;
    }

    [[nodiscard]] std::uint64_t consumed() const noexcept
    {
        return consumed_;
    }

    void finish(ArchiveBoundary boundary)
    {
        if (remaining_ != 0)
        {
            fail("trailing bytes inside the framed payload");
        }
        std::array<std::byte, 8> footer;
        read_exact(footer);
        if (!std::equal(end_marker.begin(), end_marker.end(), footer.begin()))
        {
            fail("invalid archive footer");
        }
        checksum_.update(end_marker);
        if (load_le<std::uint32_t>(std::span{footer}.last(4)) != checksum_.value())
        {
            fail("archive checksum mismatch");
        }
        if (boundary == ArchiveBoundary::EndOfStream && input_.rdbuf()->sgetc() != std::char_traits<char>::eof())
        {
            fail("trailing bytes after archive");
        }
    }

    [[noreturn]] void fail(const std::string &message) const
    {
        throw std::invalid_argument{message + " at wire byte " + std::to_string(wire_position_) + ", payload byte " +
                                    std::to_string(consumed_)};
    }

  private:
    void read_exact(std::span<std::byte> bytes)
    {
        if (!std::in_range<std::streamsize>(bytes.size()))
        {
            throw std::length_error{"archive I/O transfer exceeds streamsize"};
        }
        try
        {
            input_.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        }
        catch (const std::ios_base::failure &)
        {
            wire_position_ += static_cast<std::uint64_t>(input_.gcount());
            throw std::ios_base::failure{"failed to read archive at wire byte " + std::to_string(wire_position_)};
        }
        wire_position_ += static_cast<std::uint64_t>(input_.gcount());
        if (!input_)
        {
            throw std::ios_base::failure{"truncated archive at wire byte " + std::to_string(wire_position_)};
        }
    }

    void load_frame()
    {
        available_ = static_cast<std::size_t>(std::min<std::uint64_t>(remaining_, buffer_.size()));
        const auto frame = std::span{buffer_}.first(available_);
        std::array<std::byte, 4> encoded_crc;
        read_exact(frame);
        read_exact(encoded_crc);
        if (crc32c(frame) != load_le<std::uint32_t>(encoded_crc))
        {
            fail("archive frame checksum mismatch");
        }
        checksum_.update(frame);
        position_ = 0;
    }

    std::istream &input_;
    std::array<std::byte, frame_bytes> buffer_;
    Checksum checksum_;
    std::uint64_t wire_position_{0};
    std::uint64_t remaining_{0};
    std::uint64_t consumed_{0};
    std::size_t position_{0};
    std::size_t available_{0};
};
} // namespace minitensor::persistence::detail::binary
