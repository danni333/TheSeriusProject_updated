#pragma once

#include "../transformer/model.h"

#include <string>

namespace serius {
namespace io {

/// Load ModelConfig from a JSON file (configs/model_*.json format).
/// Maps: hidden_size -> d_model, num_attention_heads -> n_heads,
///       num_layers -> n_layers, intermediate_size -> d_ff,
///       max_position_embeddings -> max_seq_len, vocab_size -> vocab_size.
/// Returns default ModelConfig if file missing or invalid.
model::ModelConfig load_model_config(const std::string &path);

/// Load model config from JSON string (e.g. from API).
model::ModelConfig load_model_config_from_string(const std::string &json_str);

}  // namespace io
}  // namespace serius
