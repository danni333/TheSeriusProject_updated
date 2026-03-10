#include "safetensors.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace serius {
namespace io {

// ============================================================
// SafetensorsReader
// ============================================================

SafetensorsReader::SafetensorsReader(const std::string &path) : file_(path)
{
    if (file_.size() < 8) {
        throw std::runtime_error("SafetensorsReader: file too small: " + path);
    }
    parse_header();
}

void SafetensorsReader::parse_header()
{
    const uint8_t *raw = file_.bytes();

    // Read header size (8 bytes, uint64 LE)
    uint64_t header_size = 0;
    std::memcpy(&header_size, raw, sizeof(uint64_t));

    if (8 + header_size > file_.size()) {
        throw std::runtime_error("SafetensorsReader: header size exceeds file size");
    }

    // Parse JSON header
    std::string header_str(reinterpret_cast<const char *>(raw + 8), header_size);
    auto header = nlohmann::json::parse(header_str);

    data_start_ = raw + 8 + header_size;

    for (auto &[key, value] : header.items()) {
        if (key == "__metadata__") {
            // File-level metadata
            if (value.is_object()) {
                for (auto &[mk, mv] : value.items()) {
                    metadata_[mk] = mv.is_string() ? mv.get<std::string>() : mv.dump();
                }
            }
            continue;
        }

        TensorInfo info;
        info.name = key;
        info.dtype = value.at("dtype").get<std::string>();

        for (auto &s : value.at("shape")) {
            info.shape.push_back(s.get<size_t>());
        }

        auto &offsets = value.at("data_offsets");
        info.data_offset_start = offsets[0].get<size_t>();
        info.data_offset_end = offsets[1].get<size_t>();

        tensors_[key] = std::move(info);
    }
}

std::vector<std::string> SafetensorsReader::tensor_names() const
{
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto &[name, _] : tensors_) {
        names.push_back(name);
    }
    return names;
}

const TensorInfo &SafetensorsReader::tensor_info(const std::string &name) const
{
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("SafetensorsReader: tensor not found: " + name);
    }
    return it->second;
}

bool SafetensorsReader::has_tensor(const std::string &name) const
{
    return tensors_.count(name) > 0;
}

const void *SafetensorsReader::raw_data(const std::string &name) const
{
    const auto &info = tensor_info(name);
    return data_start_ + info.data_offset_start;
}

core::Tensor SafetensorsReader::get_tensor(const std::string &name) const
{
    const auto &info = tensor_info(name);
    const uint8_t *src = data_start_ + info.data_offset_start;
    size_t byte_count = info.data_offset_end - info.data_offset_start;

    if (info.dtype == "F32") {
        // Create a tensor and copy data (we don't return a view because
        // the mmap lifetime is tied to this reader object)
        core::Tensor t(info.shape);
        if (byte_count != t.size() * sizeof(float)) {
            throw std::runtime_error("SafetensorsReader: size mismatch for tensor: " + name);
        }
        std::memcpy(t.data(), src, byte_count);
        return t;
    } else if (info.dtype == "F16") {
        // Convert FP16 to FP32
        core::Tensor t(info.shape);
        size_t n = t.size();
        if (byte_count != n * 2) {
            throw std::runtime_error("SafetensorsReader: FP16 size mismatch for tensor: " + name);
        }
        const uint16_t *fp16_data = reinterpret_cast<const uint16_t *>(src);
        float *fp32_data = t.data();
        for (size_t i = 0; i < n; i++) {
            // IEEE 754 FP16 -> FP32 conversion
            uint16_t h = fp16_data[i];
            uint32_t sign = (h & 0x8000u) << 16;
            uint32_t exponent = (h >> 10) & 0x1F;
            uint32_t mantissa = h & 0x03FF;
            uint32_t f;
            if (exponent == 0) {
                if (mantissa == 0) {
                    f = sign;
                } else {
                    // Subnormal
                    exponent = 1;
                    while (!(mantissa & 0x0400)) {
                        mantissa <<= 1;
                        exponent--;
                    }
                    mantissa &= 0x03FF;
                    f = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
                }
            } else if (exponent == 31) {
                f = sign | 0x7F800000u | (mantissa << 13);  // Inf/NaN
            } else {
                f = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
            }
            std::memcpy(&fp32_data[i], &f, sizeof(float));
        }
        return t;
    } else if (info.dtype == "BF16") {
        // Convert BF16 to FP32 (just shift left by 16)
        core::Tensor t(info.shape);
        size_t n = t.size();
        if (byte_count != n * 2) {
            throw std::runtime_error("SafetensorsReader: BF16 size mismatch for tensor: " + name);
        }
        const uint16_t *bf16_data = reinterpret_cast<const uint16_t *>(src);
        float *fp32_data = t.data();
        for (size_t i = 0; i < n; i++) {
            uint32_t f = static_cast<uint32_t>(bf16_data[i]) << 16;
            std::memcpy(&fp32_data[i], &f, sizeof(float));
        }
        return t;
    }

    throw std::runtime_error("SafetensorsReader: unsupported dtype: " + info.dtype);
}

// ============================================================
// SafetensorsWriter
// ============================================================

static std::string dtype_to_string(hal::DType dtype)
{
    switch (dtype) {
        case hal::DType::Float32: return "F32";
        case hal::DType::Float16: return "F16";
        case hal::DType::BFloat16: return "BF16";
        case hal::DType::Int8: return "I8";
        case hal::DType::Int4: return "I4";
        default: return "F32";
    }
}

void SafetensorsWriter::add_tensor(const std::string &name, const core::Tensor &tensor)
{
    PendingTensor pt;
    pt.name = name;
    pt.dtype = dtype_to_string(tensor.dtype());
    pt.shape = tensor.shape();
    pt.data = tensor.raw_data();
    pt.data_bytes = tensor.size() * tensor.element_size();
    pending_.push_back(std::move(pt));
}

void SafetensorsWriter::add_metadata(const std::string &key, const std::string &value)
{
    metadata_[key] = value;
}

void SafetensorsWriter::save(const std::string &path) const
{
    // Build header JSON
    nlohmann::json header;

    // Add metadata if present
    if (!metadata_.empty()) {
        nlohmann::json meta;
        for (const auto &[k, v] : metadata_) {
            meta[k] = v;
        }
        header["__metadata__"] = meta;
    }

    // Calculate data offsets
    size_t current_offset = 0;
    std::vector<size_t> offsets;
    for (const auto &pt : pending_) {
        offsets.push_back(current_offset);
        current_offset += pt.data_bytes;
    }

    // Build tensor entries
    for (size_t i = 0; i < pending_.size(); i++) {
        const auto &pt = pending_[i];
        nlohmann::json entry;
        entry["dtype"] = pt.dtype;
        entry["shape"] = pt.shape;
        entry["data_offsets"] = {offsets[i], offsets[i] + pt.data_bytes};
        header[pt.name] = entry;
    }

    std::string header_str = header.dump();

    // Write file
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("SafetensorsWriter: cannot open file for writing: " + path);
    }

    // Header size (uint64 LE)
    uint64_t header_size = header_str.size();
    out.write(reinterpret_cast<const char *>(&header_size), sizeof(uint64_t));

    // Header JSON
    out.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));

    // Data blob
    for (const auto &pt : pending_) {
        out.write(static_cast<const char *>(pt.data), static_cast<std::streamsize>(pt.data_bytes));
    }
}

}  // namespace io
}  // namespace serius
