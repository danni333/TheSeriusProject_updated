#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace serius {
namespace io {

/// Minimal GGUF metadata (header only; no tensor data loaded).
struct GGUFMeta
{
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;
    bool valid = false;
};

/// Read GGUF magic and metadata from file. Returns valid=false on error or wrong format.
GGUFMeta load_gguf_metadata(const std::string &path);

/// List tensor names from GGUF (requires scanning metadata). Returns empty on error.
std::vector<std::string> gguf_tensor_names(const std::string &path);

}  // namespace io
}  // namespace serius
