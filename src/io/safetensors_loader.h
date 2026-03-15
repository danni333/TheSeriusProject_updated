#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace serius {
namespace io {

/// Metadata for one tensor in a safetensors file.
struct SafetensorTensorMeta
{
    std::string dtype;   // "F32", "F16", "BF16", etc.
    std::vector<int64_t> shape;
    int64_t data_start;  // byte offset in data section
    int64_t data_end;    // byte offset (exclusive)
};

/// Loaded safetensors file: header metadata and raw data block.
/// Data is kept in memory; use tensor_data() to get float pointer for a tensor.
class SafetensorsFile
{
   public:
    SafetensorsFile() = default;
    ~SafetensorsFile() = default;
    SafetensorsFile(SafetensorsFile &&) = default;
    SafetensorsFile &operator=(SafetensorsFile &&) = default;

    bool valid() const { return valid_; }
    const std::unordered_map<std::string, SafetensorTensorMeta> &tensors() const { return tensors_; }

    /// Get pointer to tensor data as float (F32). Converts F16 in-place if needed.
    /// Returns nullptr if tensor not found or dtype not supported. Lifetime: until file is destroyed.
    float *tensor_data(const std::string &name);

    /// Copy tensor into a pre-allocated float buffer (size = product(shape)). Converts F16/BF16 to F32.
    /// Returns true if copied, false if not found or wrong size.
    bool copy_tensor_to_float(const std::string &name, float *out, size_t out_size);

    /// Total number of tensors.
    size_t num_tensors() const { return tensors_.size(); }

   private:
    bool valid_ = false;
    std::unordered_map<std::string, SafetensorTensorMeta> tensors_;
    std::vector<uint8_t> data_;

    friend bool load_safetensors(const std::string &path, SafetensorsFile *out);
};

/// Load a single .safetensors file. Supports F32 and F16 (converted to F32 when accessed).
/// Returns true on success.
bool load_safetensors(const std::string &path, SafetensorsFile *out);

}  // namespace io
}  // namespace serius
