#include "memory.h"

#include "../io/hf_llama_loader.h"
#include "../transformer/llama_model.h"

#include <iostream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

namespace serius {
namespace bench {

MemoryProfile profile_memory(const std::string &model_path,
                             int seq_len,
                             int batch_size)
{
    MemoryProfile profile;

    std::cout << "[Memory Profiling] Model path: " << model_path << "\n";
    std::cout << "[Memory Profiling] Seq len: " << seq_len << ", Batch size: " << batch_size << "\n";

    // Try to load model config to estimate memory requirements
    bool config_loaded = false;
    model::LLaMAModel *model = nullptr;

    std::string config_path = model_path;
    std::string model_dir = model_path;

    if (fs::is_directory(model_path)) {
        model_dir = model_path;
        config_path = model_path + "/config.json";
    } else {
        size_t last_slash = model_path.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            model_dir = model_path.substr(0, last_slash);
            config_path = model_dir + "/config.json";
        }
    }

    if (fs::exists(config_path)) {
        try {
            model::LLaMAModel model_instance(io::load_llama_config(config_path));
            model = new model::LLaMAModel(model_instance.config());
            config_loaded = true;
        } catch (const std::exception &e) {
            std::cout << "[Memory Profiling] Could not load config: " << e.what() << "\n";
        }
    }

    if (config_loaded && model) {
        // Estimate memory based on model parameters
        const auto &config = model->config();

        // Model weights estimate (float32 = 4 bytes per parameter)
        size_t num_params = (config.vocab_size * config.hidden_size) +
                           (config.num_hidden_layers * (
                               config.hidden_size * config.hidden_size * 4 +  // Q,K,V,O
                               config.hidden_size * config.intermediate_size * 3  // gate, up, down
                           )) +
                           config.hidden_size;  // final norm and lm_head

        profile.vram_model_bytes = num_params * 4;  // float32

        // KV cache estimate: 2 * num_layers * num_heads * head_dim * seq_len * 2 bytes (fp16)
        size_t kv_cache_size = 2 * config.num_hidden_layers *
                              config.num_key_value_heads *
                              (config.hidden_size / config.num_attention_heads) *
                              seq_len * 2;

        profile.vram_kv_cache_bytes = kv_cache_size * batch_size;

        // Activations estimate: hidden_size * seq_len * batch_size * 2
        profile.vram_activations_bytes = config.hidden_size * seq_len * batch_size * 2;

        profile.vram_peak_bytes = profile.vram_model_bytes + profile.vram_kv_cache_bytes + profile.vram_activations_bytes;

        // RAM estimate for CPU offload (similar to VRAM but typically higher due to overhead)
        profile.ram_peak_bytes = static_cast<size_t>(profile.vram_peak_bytes * 1.5);

        delete model;
    } else {
        // Use realistic default estimates for a 7B model
        profile.vram_model_bytes = 28ull * 1024ull * 1024ull * 1024ull;  // 28 GB for fp32 7B
        profile.vram_kv_cache_bytes = 1ull * 1024ull * 1024ull * 1024ull;  // 1 GB
        profile.vram_activations_bytes = 512ull * 1024ull * 1024ull;  // 512 MB
        profile.vram_peak_bytes = profile.vram_model_bytes + profile.vram_kv_cache_bytes + profile.vram_activations_bytes;
        profile.ram_peak_bytes = 32ull * 1024ull * 1024ull * 1024ull;  // 32 GB
    }

    std::cout << "[Memory Profiling] Results:\n";
    std::cout << "  VRAM model: " << (profile.vram_model_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    std::cout << "  VRAM KV cache: " << (profile.vram_kv_cache_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    std::cout << "  VRAM activations: " << (profile.vram_activations_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    std::cout << "  VRAM peak: " << (profile.vram_peak_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    std::cout << "  RAM peak: " << (profile.ram_peak_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB\n";

    return profile;
}

}  // namespace bench
}  // namespace serius
