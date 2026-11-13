#pragma once

#include <string>

namespace serius {
namespace io {

/// Export a GGUF model file as a HuggingFace-style directory that the
/// safetensors loader understands: `model.safetensors` (FP32, dequantized),
/// `config.json`, and copies of `tokenizer.json`/`tokenizer_config.json`
/// when present next to the GGUF.
///
/// Tensor names are mapped from GGUF (`blk.N.attn_q.weight`, ...) to the HF
/// layout (`model.layers.N.self_attn.q_proj.weight`, ...). For the "llama"
/// architecture, Q/K projection rows are un-permuted to undo the RoPE
/// head-dim interleaving stored in GGUF files.
///
/// Returns true on success; on failure returns false and logs to stderr.
bool export_gguf_to_hf_dir(const std::string &gguf_path, const std::string &out_dir);

}  // namespace io
}  // namespace serius
