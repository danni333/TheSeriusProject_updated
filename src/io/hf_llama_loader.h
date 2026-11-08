#pragma once

#include <string>

namespace serius {
namespace model {
struct LLamaConfig;
class LLaMAModel;
}  // namespace model

namespace io {

/// Load LLaMA config from HuggingFace config.json (num_hidden_layers, hidden_size, etc.).
model::LLamaConfig load_llama_config(const std::string &config_path);

/// Load model weights from a directory. Automatically detects:
///   1. GGUF files (*.gguf) – supports Q4_0, Q4_1, Q8_0, F16, F32 quantisations.
///   2. SafeTensors shards (*.safetensors) – used as fallback.
/// Returns true if at least one file was loaded successfully.
bool load_llama_weights_from_dir(const std::string &dir_path, model::LLaMAModel *model);

/// Load from a single .safetensors file (e.g. model.safetensors).
bool load_llama_weights_from_file(const std::string &safetensors_path, model::LLaMAModel *model);

/// Load from a single .gguf file, dequantising weights on-the-fly.
/// Supports GGML Q4_0, Q4_1, Q8_0, F16, F32.
bool load_llama_weights_from_gguf_file(const std::string &gguf_path, model::LLaMAModel *model);

}  // namespace io
}  // namespace serius
