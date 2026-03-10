#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace serius {
namespace io {

/// Cross-platform memory-mapped file wrapper.
/// Maps the entire file into memory for zero-copy access.
class MappedFile
{
   public:
    MappedFile() = default;

    explicit MappedFile(const std::string &path)
    {
        open(path);
    }

    ~MappedFile()
    {
        close();
    }

    // Non-copyable
    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;

    // Movable
    MappedFile(MappedFile &&other) noexcept
        : data_(other.data_), size_(other.size_)
#ifdef _WIN32
          ,
          file_handle_(other.file_handle_), mapping_handle_(other.mapping_handle_)
#else
          ,
          fd_(other.fd_)
#endif
    {
        other.data_ = nullptr;
        other.size_ = 0;
#ifdef _WIN32
        other.file_handle_ = INVALID_HANDLE_VALUE;
        other.mapping_handle_ = nullptr;
#else
        other.fd_ = -1;
#endif
    }

    MappedFile &operator=(MappedFile &&other) noexcept
    {
        if (this != &other) {
            close();
            data_ = other.data_;
            size_ = other.size_;
#ifdef _WIN32
            file_handle_ = other.file_handle_;
            mapping_handle_ = other.mapping_handle_;
            other.file_handle_ = INVALID_HANDLE_VALUE;
            other.mapping_handle_ = nullptr;
#else
            fd_ = other.fd_;
            other.fd_ = -1;
#endif
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void open(const std::string &path)
    {
        close();
#ifdef _WIN32
        file_handle_ =
            CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("MappedFile: cannot open file: " + path);
        }
        LARGE_INTEGER li;
        GetFileSizeEx(file_handle_, &li);
        size_ = static_cast<size_t>(li.QuadPart);
        mapping_handle_ = CreateFileMapping(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_handle_) {
            CloseHandle(file_handle_);
            throw std::runtime_error("MappedFile: CreateFileMapping failed: " + path);
        }
        data_ = MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);
        if (!data_) {
            CloseHandle(mapping_handle_);
            CloseHandle(file_handle_);
            throw std::runtime_error("MappedFile: MapViewOfFile failed: " + path);
        }
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("MappedFile: cannot open file: " + path);
        }
        struct stat st;
        if (fstat(fd_, &st) != 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("MappedFile: fstat failed: " + path);
        }
        size_ = static_cast<size_t>(st.st_size);
        if (size_ == 0) {
            // Empty file: valid but no mapping needed
            data_ = nullptr;
            return;
        }
        data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("MappedFile: mmap failed: " + path);
        }
#endif
    }

    void close()
    {
#ifdef _WIN32
        if (data_) UnmapViewOfFile(data_);
        if (mapping_handle_) CloseHandle(mapping_handle_);
        if (file_handle_ != INVALID_HANDLE_VALUE) CloseHandle(file_handle_);
        data_ = nullptr;
        mapping_handle_ = nullptr;
        file_handle_ = INVALID_HANDLE_VALUE;
#else
        if (data_ && size_ > 0) munmap(data_, size_);
        if (fd_ >= 0) ::close(fd_);
        data_ = nullptr;
        fd_ = -1;
#endif
        size_ = 0;
    }

    const void *data() const { return data_; }
    const uint8_t *bytes() const { return static_cast<const uint8_t *>(data_); }
    size_t size() const { return size_; }
    bool is_open() const { return data_ != nullptr || (size_ == 0); }

   private:
    void *data_ = nullptr;
    size_t size_ = 0;
#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

}  // namespace io
}  // namespace serius
