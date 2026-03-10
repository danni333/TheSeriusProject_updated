#include "gguf_loader.h"

#include <fstream>

namespace serius {
namespace io {

namespace {

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

    // Validate version: must be in a supported range
    if (h.version < 1 || h.version > 10) {
        return out;
    }
    // Validate tensor counts: must be reasonable positive values
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
        uint64_t name_len = 0;
        f.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
        if (!f) {
            break;
        }
        std::string name(static_cast<size_t>(name_len), '\0');
        f.read(&name[0], static_cast<std::streamsize>(name_len));
        if (!f) {
            break;
        }
        names.push_back(name);
        uint32_t n_dims = 0;
        f.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        if (!f) {
            break;
        }
        for (uint32_t d = 0; d < n_dims && f; ++d) {
            uint64_t dim = 0;
            f.read(reinterpret_cast<char *>(&dim), sizeof(dim));
        }
        uint32_t dtype = 0;
        f.read(reinterpret_cast<char *>(&dtype), sizeof(dtype));
        uint64_t offset = 0;
        f.read(reinterpret_cast<char *>(&offset), sizeof(offset));
    }
    return names;
}

}  // namespace io
}  // namespace serius
