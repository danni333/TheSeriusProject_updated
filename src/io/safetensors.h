#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../hal/device.h"
#include "../tensor/tensor.h"
#include "mmap.h"

namespace serius {
namespace io {

/// Metadata for a single tensor stored in a safetensors file.
struct TensorInfo
{
    std::string name;
    std::string dtype;                      // "F32", "F16", "BF16", "I32", etc.
    std::vector<size_t> shape;
    size_t data_offset_start = 0;           // byte offset into data blob
    size_t data_offset_end = 0;             // byte offset end
};

/// Reader for the Safetensors format (mmap-based, zero-copy).
///
/// File layout:
///   [8 bytes: header_size as uint64 LE]
///   [header_size bytes: JSON header]
///   [data blob: raw tensor data]
class SafetensorsReader
{
   public:
    SafetensorsReader() = default;

    /// Open and parse a safetensors file.
    explicit SafetensorsReader(const std::string &path);

    /// Get the list of all tensor names in the file.
    std::vector<std::string> tensor_names() const;

    /// Get metadata for a specific tensor.
    const TensorInfo &tensor_info(const std::string &name) const;

    /// Check if a tensor exists.
    bool has_tensor(const std::string &name) const;

    /// Get a tensor by name. Returns a view (zero-copy) for Float32 tensors,
    /// or a converted copy for other dtypes.
    core::Tensor get_tensor(const std::string &name) const;

    /// Get raw pointer to tensor data (zero-copy).
    const void *raw_data(const std::string &name) const;

    /// Get file-level metadata (the __metadata__ key if present).
    const std::map<std::string, std::string> &metadata() const { return metadata_; }

    /// Total number of tensors in the file.
    size_t num_tensors() const { return tensors_.size(); }

   private:
    MappedFile file_;
    std::map<std::string, TensorInfo> tensors_;
    std::map<std::string, std::string> metadata_;
    const uint8_t *data_start_ = nullptr;

    void parse_header();
};

/// Writer for the Safetensors format.
class SafetensorsWriter
{
   public:
    /// Add a tensor to be written.
    void add_tensor(const std::string &name, const core::Tensor &tensor);

    /// Add file-level metadata.
    void add_metadata(const std::string &key, const std::string &value);

    /// Write all tensors to a file.
    void save(const std::string &path) const;

   private:
    struct PendingTensor
    {
        std::string name;
        std::string dtype;
        std::vector<size_t> shape;
        const void *data;
        size_t data_bytes;
    };

    std::vector<PendingTensor> pending_;
    std::map<std::string, std::string> metadata_;
};

}  // namespace io
}  // namespace serius
