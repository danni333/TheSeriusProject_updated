#include "safetensors_loader.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include <nlohmann/json.hpp>

namespace serius {
namespace io {

namespace {

template <typename T>
T read_le(std::istream &f)
{
    T v{};
    f.read(reinterpret_cast<char *>(&v), sizeof(T));
    return v;
}

bool parse_header(const std::string &json_str, std::unordered_map<std::string, SafetensorTensorMeta> *out)
{
    try {
        auto j = nlohmann::json::parse(json_str);
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (it.key() == "__metadata__") continue;
            const auto &v = it.value();
            if (!v.is_object() || !v.contains("dtype") || !v.contains("shape") || !v.contains("data_offsets"))
                continue;
            SafetensorTensorMeta m;
            m.dtype = v["dtype"].get<std::string>();
            for (const auto &s : v["shape"]) m.shape.push_back(s.get<int64_t>());
            const auto &off = v["data_offsets"];
            m.data_start = off[0].get<int64_t>();
            m.data_end = off[1].get<int64_t>();
            (*out)[it.key()] = std::move(m);
        }
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

void f16_to_f32(const uint16_t *src, float *dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        uint16_t h = src[i];
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0) {
            float val = (mant == 0) ? 0.0f : std::ldexp(static_cast<float>(mant) / 1024.0f, -14);
            dst[i] = sign ? -val : val;
        } else if (exp == 31) {
            dst[i] = (mant == 0) ? (sign ? -INFINITY : INFINITY) : std::numeric_limits<float>::quiet_NaN();
        } else {
            float val = std::ldexp(1.0f + static_cast<float>(mant) / 1024.0f, exp - 15);
            dst[i] = sign ? -val : val;
        }
    }
}

}  // namespace

bool load_safetensors(const std::string &path, SafetensorsFile *out)
{
    if (!out) return false;
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    uint64_t header_size = read_le<uint64_t>(f);
    if (header_size == 0 || header_size > 100 * 1024 * 1024) return false;

    std::string header_json(static_cast<size_t>(header_size), '\0');
    f.read(&header_json[0], static_cast<std::streamsize>(header_size));
    if (!f || f.gcount() != static_cast<std::streamsize>(header_size)) return false;

    out->tensors_.clear();
    if (!parse_header(header_json, &out->tensors_)) return false;

    f.seekg(0, std::ios::end);
    std::streamsize file_size = f.tellg();
    std::streamsize data_offset = 8 + static_cast<std::streamsize>(header_size);
    std::streamsize data_size = file_size - data_offset;
    if (data_size <= 0) return false;

    out->data_.resize(static_cast<size_t>(data_size));
    f.seekg(data_offset);
    f.read(reinterpret_cast<char *>(out->data_.data()), data_size);
    if (!f || f.gcount() != data_size) return false;

    out->valid_ = true;
    return true;
}

float *SafetensorsFile::tensor_data(const std::string &name)
{
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    const auto &m = it->second;
    if (m.data_start < 0 || m.data_end > static_cast<int64_t>(data_.size()) || m.data_start >= m.data_end)
        return nullptr;
    size_t n = static_cast<size_t>((m.data_end - m.data_start) / 2);  // F16
    if (m.dtype == "F32") {
        n = static_cast<size_t>((m.data_end - m.data_start) / 4);
        return reinterpret_cast<float *>(data_.data() + m.data_start);
    }
    if (m.dtype == "F16") {
        static thread_local std::vector<float> convert_buf;
        convert_buf.resize(n);
        f16_to_f32(reinterpret_cast<const uint16_t *>(data_.data() + m.data_start), convert_buf.data(), n);
        return convert_buf.data();  // Invalid after next call to tensor_data for F16!
    }
    return nullptr;
}

bool SafetensorsFile::copy_tensor_to_float(const std::string &name, float *out, size_t out_size)
{
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return false;
    const auto &m = it->second;
    size_t num_elems = 1;
    for (int64_t d : m.shape) num_elems *= static_cast<size_t>(d);
    if (num_elems != out_size) return false;

    int64_t byte_len = m.data_end - m.data_start;
    if (m.dtype == "F32") {
        if (byte_len != static_cast<int64_t>(out_size * sizeof(float))) return false;
        std::memcpy(out, data_.data() + m.data_start, static_cast<size_t>(byte_len));
        return true;
    }
    if (m.dtype == "F16") {
        if (byte_len != static_cast<int64_t>(out_size * 2)) return false;
        f16_to_f32(reinterpret_cast<const uint16_t *>(data_.data() + m.data_start), out, out_size);
        return true;
    }
    return false;
}

}  // namespace io
}  // namespace serius
