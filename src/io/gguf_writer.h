#pragma once

#include <string>

namespace serius {
namespace io {

struct GGUFWritOptions
{
    /// Store tensors as FP32 instead of FP16 (larger file, no precision loss).
    bool use_f32 = false;

    /// Override for `general.architecture` (e.g. "qwen2", "llama"). When empty
    /// the value is derived from config.json (model_type / architectures[0]).
    std::string arch;

    /// Embed tokenizer.ggml.* KV metadata parsed from tokenizer.json /
    /// merges.txt / tokenizer_config.json found in the model directory.
    bool embed_tokenizer = true;
};

/// Convert a HuggingFace-format model directory to a single GGUF file.
///
/// @param hf_dir   Directory containing config.json and model weights
///                 (model.safetensors, a *.safetensors.index.json shard
///                 manifest, or a single .safetensors file may also be
///                 passed directly — its parent dir is used).
/// @param out_path Destination .gguf path.
/// @return true on success.
bool write_hf_dir_to_gguf(const std::string &hf_dir, const std::string &out_path, const GGUFWritOptions &opt = {});

}  // namespace io
}  // namespace serius
