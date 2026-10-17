#include "gguf_loader.h"

#include <array>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>

#include "../transformer/config.h"

namespace serius {
namespace io {

namespace {

using int4_t = std::int8_t;

constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" little-endian

#pragma pack(push, 1)
struct GGUFHeader
{
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
};
#pragma pack(pop)

float half_to_float(uint16_t half_val)
{
    const uint32_t sign = (static_cast<uint32_t>(half_val) >> 15) & 0x1u;
    const uint32_t exp = (static_cast<uint32_t>(half_val) >> 10) & 0x1Fu;
    const uint32_t mant = static_cast<uint32_t>(half_val) & 0x3FFu;

    uint32_t bits = 0;
    if (exp == 0u) {
        bits = (sign << 31);
        if (mant != 0u) {
            // subnormal half -> normalize to float
            const uint32_t mantissa = mant << 13;
            bits |= mantissa;
        }
    } else if (exp == 0x1Fu) {
        bits = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void dequantize_q4_0(const uint8_t *data, size_t n, float *out)
{
    for (size_t i = 0; i < n; i += 32) {
        uint16_t scale = data[0] | (data[1] << 8);
        float scale_f = half_to_float(scale);
        const uint8_t *q = data + 2;
        for (size_t j = 0; j < 16 && i + j < n; ++j) {
            int4_t v0 = (q[j >> 1] >> 0) & 0x0F;
            int4_t v1 = (q[j >> 1] >> 4) & 0x0F;
            out[i + j * 2 + 0] = (v0 - 8) * scale_f;
            out[i + j * 2 + 1] = (v1 - 8) * scale_f;
        }
        data += 18;
    }
}

void dequantize_q4_1(const uint8_t *data, size_t n, float *out)
{
    for (size_t i = 0; i < n; i += 32) {
        uint16_t d = data[0] | (data[1] << 8);
        uint16_t m = data[2] | (data[3] << 8);
        float d_f = half_to_float(d);
        float m_f = half_to_float(m);
        const uint8_t *q = data + 4;
        for (size_t j = 0; j < 16 && i + j < n; ++j) {
            int4_t v0 = (q[j >> 1] >> 0) & 0x0F;
            int4_t v1 = (q[j >> 1] >> 4) & 0x0F;
            out[i + j * 2 + 0] = v0 * d_f + m_f;
            out[i + j * 2 + 1] = v1 * d_f + m_f;
        }
        data += 20;
    }
}

void dequantize_q8_0(const uint8_t *data, size_t n, float *out)
{
    for (size_t i = 0; i < n; i += 32) {
        uint16_t scale = data[0] | (data[1] << 8);
        float scale_f = half_to_float(scale);
        const int8_t *q = reinterpret_cast<const int8_t *>(data + 2);
        for (size_t j = 0; j < 32 && i + j < n; ++j) {
            out[i + j] = q[j] * scale_f;
        }
        data += 34;
    }
}

}  // anonymous namespace

namespace {

uint64_t read_u64_le(std::ifstream &f)
{
    uint64_t value = 0;
    std::array<uint8_t, sizeof(value)> bytes{};
    f.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    for (size_t i = 0; i < bytes.size(); ++i) {
        value |= (static_cast<uint64_t>(bytes[i]) << (8 * i));
    }
    return value;
}

uint32_t read_u32_le(std::ifstream &f)
{
    uint32_t value = 0;
    std::array<uint8_t, sizeof(value)> bytes{};
    f.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    for (size_t i = 0; i < bytes.size(); ++i) {
        value |= (static_cast<uint32_t>(bytes[i]) << (8 * i));
    }
    return value;
}

uint16_t read_u16_le(std::ifstream &f)
{
    uint16_t value = 0;
    std::array<uint8_t, sizeof(value)> bytes{};
    f.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    for (size_t i = 0; i < bytes.size(); ++i) {
        value |= (static_cast<uint16_t>(bytes[i]) << (8 * i));
    }
    return value;
}

int32_t read_i32_le(std::ifstream &f)
{
    uint32_t raw = read_u32_le(f);
    return static_cast<int32_t>(raw);
}

int64_t read_i64_le(std::ifstream &f)
{
    uint64_t raw = read_u64_le(f);
    return static_cast<int64_t>(raw);
}

bool read_string_kv_value(std::ifstream &f, std::string *out)
{
    uint64_t len = read_u64_le(f);
    if (!f) return false;
    std::string value(static_cast<size_t>(len), '\0');
    f.read(&value[0], static_cast<std::streamsize>(len));
    if (!f) return false;
    if (out) *out = std::move(value);
    return true;
}

bool skip_gguf_value(std::ifstream &f, uint32_t type)
{
    switch (type) {
        case 0: return true;
        case 1: /* uint8 */ { uint8_t v = 0; f.read(reinterpret_cast<char *>(&v), 1); return f.good(); }
        case 2: /* int8 */ { int8_t v = 0; f.read(reinterpret_cast<char *>(&v), 1); return f.good(); }
        case 3: /* uint16 */ { uint16_t v = 0; f.read(reinterpret_cast<char *>(&v), sizeof(v)); return f.good(); }
        case 4: /* uint32 */ { uint32_t v = 0; f.read(reinterpret_cast<char *>(&v), sizeof(v)); return f.good(); }
        case 5: /* int32 */ { int32_t v = 0; f.read(reinterpret_cast<char *>(&v), sizeof(v)); return f.good(); }
        case 6: /* float32 */ { float v = 0.0f; f.read(reinterpret_cast<char *>(&v), sizeof(v)); return f.good(); }
        case 7: /* bool */ { uint8_t v = 0; f.read(reinterpret_cast<char *>(&v), 1); return f.good(); }
        case 8: return read_string_kv_value(f, nullptr);
        case 9: {
            uint32_t n = read_u32_le(f);
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t elem_type = read_u32_le(f);
                if (!skip_gguf_value(f, elem_type)) return false;
            }
            return f.good();
        }
        default: {
            static const size_t type_sizes[] = {0, 1, 1, 2, 4, 4, 4, 1, 0, 0, 4, 8};
            if (type < sizeof(type_sizes) / sizeof(type_sizes[0]) && type_sizes[type] > 0) {
                f.seekg(static_cast<std::streamoff>(type_sizes[type]), std::ios::cur);
                return f.good();
            }
            return true;
        }
    }
}

bool read_gguf_kv_entry(std::ifstream &f, std::string *key, uint32_t *type, uint64_t *value_length)
{
    uint64_t key_len = read_u64_le(f);
    if (!f) return false;
    std::string key_value(static_cast<size_t>(key_len), '\0');
    f.read(&key_value[0], static_cast<std::streamsize>(key_len));
    if (!f) return false;

    *type = read_u32_le(f);
    if (!f) return false;

    if (key) *key = std::move(key_value);
    if (value_length) *value_length = 0;
    return true;
}

bool read_scalar_value(std::ifstream &f, uint32_t type, uint64_t *out)
{
    switch (type) {
        case 1: {
            uint8_t v = 0;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            if (out) *out = v;
            return f.good();
        }
        case 2: {
            int8_t v = 0;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            if (out) *out = static_cast<uint64_t>(v);
            return f.good();
        }
        case 3: {
            uint16_t v = 0;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            if (out) *out = v;
            return f.good();
        }
        case 4: {
            uint32_t v = 0;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            if (out) *out = v;
            return f.good();
        }
        case 5: {
            int32_t v = 0;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            if (out) *out = static_cast<uint64_t>(v);
            return f.good();
        }
        case 7: {
            uint8_t v = 0;
            f.read(reinterpret_cast<char *>(&v), sizeof(v));
            if (out) *out = v;
            return f.good();
        }
        case 6:
        case 8:
        case 9:
        default:
            return false;
    }
}

GGUFTensorInfo read_tensor_info(std::ifstream &f)
{
    GGUFTensorInfo info;
    uint64_t name_len = read_u64_le(f);
    if (!f) return info;
    std::string name(static_cast<size_t>(name_len), '\0');
    f.read(&name[0], static_cast<std::streamsize>(name_len));
    if (!f) return info;
    info.name = std::move(name);

    uint32_t n_dims = read_u32_le(f);
    if (!f) return info;
    info.shape.resize(n_dims);
    for (uint32_t i = 0; i < n_dims; ++i) {
        info.shape[i] = read_u64_le(f);
        if (!f) return info;
    }

    uint32_t dtype = read_u32_le(f);
    if (!f) return info;
    info.dtype = static_cast<GGMLType>(dtype);

    info.offset = read_u64_le(f);
    return info;
}

}  // namespace

GGUFMeta load_gguf_metadata(const std::string &path)
{
    GGUFMeta out;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return out;
    }

    GGUFHeader h{};
    f.read(reinterpret_cast<char *>(&h), sizeof(h));
    if (!f || h.magic != GGUF_MAGIC) {
        return out;
    }

    if (h.version < 1 || h.version > 10) {
        return out;
    }

    out.version = h.version;
    out.n_tensors = h.n_tensors;
    out.n_kv = h.n_kv;
    out.valid = true;
    return out;
}

std::vector<std::string> gguf_tensor_names(const std::string &path)
{
    std::vector<std::string> names;
    GGUFMeta meta = load_gguf_metadata(path);
    if (!meta.valid) {
        return names;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return names;
    }

    f.seekg(sizeof(GGUFHeader));
    for (uint64_t i = 0; i < meta.n_kv; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_gguf_kv_entry(f, &key, &type, nullptr)) {
            break;
        }
        (void)key;
        if (!skip_gguf_value(f, type)) {
            break;
        }
    }

    for (uint64_t i = 0; i < meta.n_tensors && f; ++i) {
        GGUFTensorInfo info = read_tensor_info(f);
        if (!f || info.name.empty()) {
            break;
        }
        names.push_back(info.name);
    }
    return names;
}

std::optional<GGUFTensorInfo> get_tensor_info(const std::string &path, const std::string &tensor_name)
{
    GGUFMeta meta = load_gguf_metadata(path);
    if (!meta.valid) {
        return std::nullopt;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return std::nullopt;
    }

    f.seekg(sizeof(GGUFHeader));
    for (uint64_t i = 0; i < meta.n_kv; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_gguf_kv_entry(f, &key, &type, nullptr)) {
            break;
        }
        (void)key;
        if (!skip_gguf_value(f, type)) {
            break;
        }
    }

    for (uint64_t i = 0; i < meta.n_tensors; ++i) {
        GGUFTensorInfo info = read_tensor_info(f);
        if (!f) {
            break;
        }
        if (info.name == tensor_name) {
            return info;
        }
    }
    return std::nullopt;
}

std::unordered_map<std::string, core::Tensor> load_all_tensors(const std::string &path)
{
    std::unordered_map<std::string, core::Tensor> result;
    GGUFMeta meta = load_gguf_metadata(path);
    if (!meta.valid) {
        return result;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return result;
    }

    f.seekg(sizeof(GGUFHeader));
    for (uint64_t i = 0; i < meta.n_kv; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_gguf_kv_entry(f, &key, &type, nullptr)) {
            break;
        }
        (void)key;
        if (!skip_gguf_value(f, type)) {
            break;
        }
    }

    std::vector<GGUFTensorInfo> tensor_infos;
    for (uint64_t i = 0; i < meta.n_tensors; ++i) {
        GGUFTensorInfo info = read_tensor_info(f);
        if (!f || info.name.empty()) {
            break;
        }
        tensor_infos.push_back(info);
    }
    if (tensor_infos.empty()) {
        return result;
    }

    const std::streampos data_base = f.tellg();
    for (const auto &info : tensor_infos) {
        std::vector<size_t> shape;
        shape.reserve(info.shape.size());
        for (uint64_t dim : info.shape) {
            shape.push_back(static_cast<size_t>(dim));
        }

        const size_t numel = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
        std::vector<float> values(numel, 0.0f);
        f.seekg(data_base + static_cast<std::streamoff>(info.offset));

        if (info.dtype == GGMLType::F32) {
            f.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(numel * sizeof(float)));
        } else if (info.dtype == GGMLType::F16) {
            std::vector<uint16_t> half_values(numel);
            f.read(reinterpret_cast<char *>(half_values.data()), static_cast<std::streamsize>(numel * sizeof(uint16_t)));
            for (size_t i = 0; i < numel; ++i) {
                values[i] = half_to_float(half_values[i]);
            }
        } else if (info.dtype == GGMLType::Q4_0) {
            const size_t block_count = (numel + 31) / 32;
            std::vector<uint8_t> packed(block_count * 18);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_q4_0(packed.data(), numel, values.data());
        } else if (info.dtype == GGMLType::Q4_1) {
            const size_t block_count = (numel + 31) / 32;
            std::vector<uint8_t> packed(block_count * 20);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_q4_1(packed.data(), numel, values.data());
        } else if (info.dtype == GGMLType::Q8_0) {
            const size_t block_count = (numel + 31) / 32;
            std::vector<uint8_t> packed(block_count * 34);
            f.read(reinterpret_cast<char *>(packed.data()), static_cast<std::streamsize>(packed.size()));
            dequantize_q8_0(packed.data(), numel, values.data());
        } else {
            std::fill(values.begin(), values.end(), 0.0f);
        }

        core::Tensor tensor(shape, hal::DType::Float32);
        for (size_t i = 0; i < values.size(); ++i) {
            tensor[{i}] = values[i];
        }
        result.emplace(info.name, std::move(tensor));
    }

    return result;
}

serius::model::ModelConfig gguf_to_model_config(const std::string &path)
{
    serius::model::ModelConfig config;
    GGUFMeta meta = load_gguf_metadata(path);
    if (!meta.valid) {
        return config;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return config;
    }

    f.seekg(sizeof(GGUFHeader));
    for (uint64_t i = 0; i < meta.n_kv; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_gguf_kv_entry(f, &key, &type, nullptr)) {
            break;
        }

        uint64_t value = 0;
        if (read_scalar_value(f, type, &value)) {
            if (key == "llama.vocab_size") {
                config.vocab_size = static_cast<size_t>(value);
            } else if (key == "llama.embedding_length") {
                config.d_model = static_cast<size_t>(value);
            } else if (key == "llama.block_count") {
                config.n_layers = static_cast<size_t>(value);
            } else if (key == "llama.attention.head_count") {
                config.n_heads = static_cast<size_t>(value);
                config.n_kv_heads = static_cast<size_t>(value);
            } else if (key == "llama.attention.head_count_kv") {
                config.n_kv_heads = static_cast<size_t>(value);
            } else if (key == "llama.feed_forward_length") {
                config.d_ff = static_cast<size_t>(value);
            } else if (key == "llama.context_length") {
                config.max_seq_len = static_cast<size_t>(value);
            }
        } else {
            if (!skip_gguf_value(f, type)) {
                break;
            }
        }
    }

    return config;
}

}  // namespace io
}  // namespace serius
