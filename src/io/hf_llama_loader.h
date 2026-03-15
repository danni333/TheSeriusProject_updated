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

/// Load model weights from one or more .safetensors files in a directory.
/// Expects files named model-00001-of-00002.safetensors or model.safetensors.
/// Weights are copied into the model (HF uses [out_features, in_features], we transpose to [in, out]).
/// Returns true if all expected tensors were found and copied.
bool load_llama_weights_from_dir(const std::string &dir_path, model::LLaMAModel *model);

/// Load from a single .safetensors file (e.g. model.safetensors).
bool load_llama_weights_from_file(const std::string &safetensors_path, model::LLaMAModel *model);

}  // namespace io
}  // namespace serius
