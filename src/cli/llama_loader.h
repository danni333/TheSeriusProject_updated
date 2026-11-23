#pragma once

#include <memory>
#include <string>

#include "tokenizer/bpe.h"
#include "transformer/llama_model.h"

namespace serius {
namespace cli {

/// Result of load_llama_cli_model: a ready-to-use LLaMAModel plus its
/// tokenizer, after hf:// resolution, config/weight loading and GPU offload.
struct LlamaCliModel
{
    std::unique_ptr<model::LLaMAModel> model;
    BPETokenizer tokenizer;
    std::string resolved_path;  ///< local path after hf:// resolution
    bool weights_loaded = false;
};

/// Load a LLaMA model for CLI commands (chat, eval, ...).
/// Handles hf://repo_id resolution, tokenizer.json / GGUF tokenizer loading,
/// config.json or GGUF-metadata config, safetensors-dir or GGUF weights, and
/// GPU offload decision (--gpu auto/cpu/cuda).
/// `gpu_layers` > 0 requests partial offload of that many leading layers
/// (--gpu-layers N); <= 0 keeps the auto/cpu/cuda decision.
/// Returns nullptr on failure (reason printed to stderr).
std::unique_ptr<LlamaCliModel> load_llama_cli_model(const std::string &model_path,
                                                    const std::string &gpu_mode,
                                                    const std::string &quant_tag = "",
                                                    int gpu_layers = -1);

}  // namespace cli
}  // namespace serius
