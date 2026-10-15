#include "gguf_loader.h"

#include <fstream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>
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

struct KVEntry
{
    std::string key;
    uint32_t type;
};

std::vector<KVEntry> read_kv_entries(std::ifstream &f, uint64_t n_kv)
{
    std::vector<KVEntry> entries;
    entries.reserve(n_kv);
    for (uint64_t i = 0; i < n_kv; ++i) {
        uint64_t key_len = 0;
        f.read(reinterpret_cast<char *>(&key_len), sizeof(key_len));
        if (!f) break;
        std::string key(static_cast<size_t>(key_len), '\0');
        f.read(&key[0], static_cast<std::streamsize>(key_len));
        if (!f) break;
        uint32_t type = 0;
        f.read(reinterpret_cast<char *>(&type), sizeof(type));
        if (!f) break;
        
        entries.push_back({key, type});
        
        if (type == 8) {  // string
            uint64_t len = 0;
            f.read(reinterpret_cast<char *>(&len), sizeof(len));
            f.seekg(f.tellg() + static_cast<std::streamoff>(len));
        } else if (type == 9) {  // array
            uint32_t n = 0;
            f.read(reinterpret_cast<char *>(&n), sizeof(n));
            for (uint32_t j = 0; j < n && f; ++j) {
                uint64_t elem_len = 0;
                f.read(reinterpret_cast<char *>(&elem_len), sizeof(elem_len));
                f.seekg(f.tellg() + static_cast<std::streamoff>(elem_len));
            }
        } else {
            static const size_t type_sizes[] = {0, 1, 1, 2, 2, 4, 4, 8, 0, 0, 4, 8};
            if (type < sizeof(type_sizes) / sizeof(type_sizes[0]) && type_sizes[type] > 0) {
                f.seekg(f.tellg() + static_cast<std::streamoff>(type_sizes[type]));
            }
        }
    }
    return entries;
}

GGUFTensorInfo read_tensor_info(std::ifstream &f)
{
    GGUFTensorInfo info;
    uint64_t name_len = 0;
    f.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
    if (!f) return info;
    info.name.resize(static_cast<size_t>(name_len));
    f.read(&info.name[0], static_cast<std::streamsize>(name_len));
    if (!f) return info;
    
    uint32_t n_dims = 0;
    f.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
    info.shape.resize(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
        f.read(reinterpret_cast<char *>(&info.shape[d]), sizeof(info.shape[d]));
    }
    
    uint32_t dtype = 0;
    f.read(reinterpret_cast<char *>(&dtype), sizeof(dtype));
    info.dtype = static_cast<GGMLType>(dtype);
    
    f.read(reinterpret_cast<char *>(&info.offset), sizeof(info.offset));
    return info;
}

bool read_string_kv(std::ifstream &f, const std::string &key, std::string &out_value)
{
    (void)key;
    uint32_t type = 0;
    f.read(reinterpret_cast<char *>(&type), sizeof(type));
    if (!f || type != 8) return false;
    uint64_t len = 0;
    f.read(reinterpret_cast<char *>(&len), sizeof(len));
    out_value.resize(static_cast<size_t>(len));
    f.read(&out_value[0], static_cast<std::streamsize>(len));
    return f.good();
}

bool read_uint_kv(std::ifstream &f, const std::string &key, uint64_t &out_value)
{
    (void)key;
    uint32_t type = 0;
    f.read(reinterpret_cast<char *>(&type), sizeof(type));
    if (!f) return false;
    if (type == 4) {  // uint32
        uint32_t v = 0;
        f.read(reinterpret_cast<char *>(&v), sizeof(v));
        out_value = v;
        return f.good();
    }
    if (type == 5) {  // uint64
        f.read(reinterpret_cast<char *>(&out_value), sizeof(out_value));
        return f.good();
    }
    return false;
}

}  // anonymous namespace

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
    if (h.n_tensors == 0 || h.n_tensors > 1000000) {
        return out;
    }
    if (h.n_kv == 0 || h.n_kv > 1000000) {
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
        uint64_t key_len = 0;
        f.read(reinterpret_cast<char *>(&key_len), sizeof(key_len));
        if (!f) {
            break;
        }
        std::string key(static_cast<size_t>(key_len), '\0');
        f.read(&key[0], static_cast<std::streamsize>(key_len));
        if (!f) {
            break;
        }
        uint32_t type = 0;
        f.read(reinterpret_cast<char *>(&type), sizeof(type));
        if (!f) {
            break;
        }
        if (type == 8) {
            uint64_t len = 0;
            f.read(reinterpret_cast<char *>(&len), sizeof(len));
            f.seekg(f.tellg() + static_cast<std::streamoff>(len));
        } else if (type == 9) {
            uint32_t n = 0;
            f.read(reinterpret_cast<char *>(&n), sizeof(n));
            for (uint32_t j = 0; j < n && f; ++j) {
                uint64_t elem_len = 0;
                f.read(reinterpret_cast<char *>(&elem_len), sizeof(elem_len));
                f.seekg(f.tellg() + static_cast<std::streamoff>(elem_len));
            }
        } else {
            static const size_t type_sizes[] = {0, 1, 1, 2, 2, 4, 4, 8, 0, 0, 4, 8};
            if (type < sizeof(type_sizes) / sizeof(type_sizes[0]) && type_sizes[type] > 0) {
                f.seekg(f.tellg() + static_cast<std::streamoff>(type_sizes[type]));
            }
        }
    }
    for (uint64_t i = 0; i < meta.n_tensors && f; ++i) {
        GGUFTensorInfo info = read_tensor_info(f);
        if (!f) break;
        names.push_back(info.name);
    }
    return names;
}

std::optional<GGUFTensorInfo> get_tensor_info(const std::string &path, const std::string &tensor_name)
{
    GGUFMeta meta = load_gguf_metadata(path);
    if (!meta.valid) return std::nullopt;
    
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::nullopt;
    
    f.seekg(sizeof(GGUFHeader));
    read_kv_entries(f, meta.n_kv);
    
    for (uint64_t i = 0; i < meta.n_tensors; ++i) {
        GGUFTensorInfo info = read_tensor_info(f);
        if (!f) break;
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
    if (!meta.valid) return result;
    
    std::ifstream f(path, std::ios::binary);
    if (!f) return result;
    
    f.seekg(sizeof(GGUFHeader));
    read_kv_entries(f, meta.n_kv);
    
    std::vector<GGUFTensorInfo> tensor_infos;
    for (uint64_t i = 0; i < meta.n_tensors; ++i) {
        GGUFTensorInfo info = read_tensor_info(f);
        if (!f) break;
        tensor_infos.push_back(info);
    }
    
    size_t data_start = static_cast<size_t>(f.tellg());
    constexpr size_t alignment = 32;
    size_t aligned_data_start = ((data_start + alignment - 1) / alignment) * alignment;
    
    size_t total_elements = 0;
    for (const auto &info : tensor_infos) {
        size_t n = 1;
        for (uint64_t d : info.shape) n *= d;
        total_elements += n;
    }
    
    std::vector<float> all_data;
    all_data.reserve(total_elements);
    
    for (const auto &info : tensor_infos) {
        size_t n = 1;
        for (uint64_t d : info.shape) n *= d;
        
        std::vector<float> tensor_data(n);
        f.seekg(static_cast<std::streamoff>(aligned_data_start + info.offset));
        
        if (info.dtype == GGMLType::F32) {
            f.read(reinterpret_cast<char *>(tensor_data.data()), n * sizeof(float));
        } else if (info.dtype == GGMLType::F16) {
            std::vector<uint16_t> half_data(n);
            f.read(reinterpret_cast<char *>(half_data.data()), n * sizeof(uint16_t));
            for (size_t i = 0; i < n; ++i) {
                tensor_data[i] = half_to_float(half_data[i]);
            }
        } else if (info.dtype == GGMLType::Q4_0) {
            size_t blocks = (n + 31) / 32;
            std::vector<uint8_t> qdata(blocks * 18);
            f.read(reinterpret_cast<char *>(qdata.data()), qdata.size());
            dequantize_q4_0(qdata.data(), n, tensor_data.data());
        } else if (info.dtype == GGMLType::Q4_1) {
            size_t blocks = (n + 31) / 32;
            std::vector<uint8_t> qdata(blocks * 20);
            f.read(reinterpret_cast<char *>(qdata.data()), qdata.size());
            dequantize_q4_1(qdata.data(), n, tensor_data.data());
        } else if (info.dtype == GGMLType::Q8_0) {
            size_t blocks = (n + 31) / 32;
            std::vector<uint8_t> qdata(blocks * 34);
            f.read(reinterpret_cast<char *>(qdata.data()), qdata.size());
            dequantize_q8_0(qdata.data(), n, tensor_data.data());
        } else {
            std::fill(tensor_data.begin(), tensor_data.end(), 0.0f);
        }
        
        std::vector<size_t> shape;
        for (uint64_t d : info.shape) {
            shape.push_back(static_cast<size_t>(d));
        }
        core::Tensor tensor(shape, hal::DType::Float32);
        for (size_t i = 0; i < n; ++i) {
            tensor[{i}] = tensor_data[i];
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
    std::vector<KVEntry> kv_entries = read_kv_entries(f, meta.n_kv);

    for (const auto &entry : kv_entries) {
        if (entry.key == "llama.vocab_size") {
            uint64_t val = 0;
            f.seekg(f.tellg() - static_cast<std::streamoff>(sizeof(uint32_t)));
            if (read_uint_kv(f, entry.key, val)) {
                config.vocab_size = static_cast<size_t>(val);
            }
        } else if (entry.key == "llama.embedding_length") {
            uint64_t val = 0;
            f.seekg(f.tellg() - static_cast<std::streamoff>(sizeof(uint32_t)));
            if (read_uint_kv(f, entry.key, val)) {
                config.d_model = static_cast<size_t>(val);
            }
        } else if (entry.key == "llama.block_count") {
            uint64_t val = 0;
            f.seekg(f.tellg() - static_cast<std::streamoff>(sizeof(uint32_t)));
            if (read_uint_kv(f, entry.key, val)) {
                config.n_layers = static_cast<size_t>(val);
            }
        } else if (entry.key == "llama.attention.head_count") {
            uint64_t val = 0;
            f.seekg(f.tellg() - static_cast<std::streamoff>(sizeof(uint32_t)));
            if (read_uint_kv(f, entry.key, val)) {
                config.n_heads = static_cast<size_t>(val);
                config.n_kv_heads = static_cast<size_t>(val);
            }
        } else if (entry.key == "llama.attention.head_count_kv") {
            uint64_t val = 0;
            f.seekg(f.tellg() - static_cast<std::streamoff>(sizeof(uint32_t)));
            if (read_uint_kv(f, entry.key, val)) {
                config.n_kv_heads = static_cast<size_t>(val);
            }
        } else if (entry.key == "llama.feed_forward_length") {
            uint64_t val = 0;
            f.seekg(f.tellg() - static_cast<std::streamoff>(sizeof(uint32_t)));
            if (read_uint_kv(f, entry.key, val)) {
                config.d_ff = static_cast<size_t>(val);
            }
        } else if (entry.key == "llama.context_length") {
            uint64_t val = 0;
            f.seekg(f.tellg() - static_cast<std::streamoff>(sizeof(uint32_t)));
            if (read_uint_kv(f, entry.key, val)) {
                config.max_seq_len = static_cast<size_t>(val);
            }
        }
    }

    return config;
}

}  // namespace io
}  // namespace serius
