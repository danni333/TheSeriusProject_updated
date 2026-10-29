#pragma once

#include <cstddef>
#include <string>

namespace serius {
namespace bench {

struct MemoryProfile
{
    size_t vram_model_bytes = 0;      // Model weights in VRAM
    size_t vram_kv_cache_bytes = 0;   // KV cache allocated
    size_t vram_activations_bytes = 0; // Temporary activations
    size_t vram_peak_bytes = 0;       // Peak total VRAM
    size_t ram_peak_bytes = 0;        // Peak system RAM
};

/// Profile memory usage during model loading and inference.
/// Measures VRAM and RAM consumption at different stages.
MemoryProfile profile_memory(const std::string &model_path,
                             int seq_len = 512,
                             int batch_size = 1);

}  // namespace bench
}  // namespace serius
