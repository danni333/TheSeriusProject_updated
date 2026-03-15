#pragma once

#include "../tensor/tensor.h"
#include "../transformer/llama_model.h"

namespace serius {
namespace infer {

struct KVCacheLayer {
    KVCacheLayer() : k({0, 0}), v({0, 0}), current_len(0) {}
    core::Tensor k; // [max_seq_len, kv_dim]
    core::Tensor v; // [max_seq_len, kv_dim]
    size_t current_len = 0;
};

struct KVCache {
    KVCache() 
        : scratch_h({0, 0}), scratch_normed({0, 0}), scratch_q({0, 0}), 
          scratch_k({0, 0}), scratch_v({0, 0}), scratch_attn_concat({0, 0}),
          scratch_mlp_gate({0, 0}), scratch_mlp_up({0, 0}), scratch_mlp_down({0, 0}) {}

    std::vector<KVCacheLayer> layers;
    
    // Scratch buffers to avoid repeated allocations during inference
    core::Tensor scratch_h;
    core::Tensor scratch_normed;
    core::Tensor scratch_q;
    core::Tensor scratch_k;
    core::Tensor scratch_v;
    core::Tensor scratch_attn_concat;
    core::Tensor scratch_mlp_gate;
    core::Tensor scratch_mlp_up;
    core::Tensor scratch_mlp_down;
    std::vector<float> scratch_scores; // [n_heads * max_seq_len] for thread-safety

    void clear() {
        for (auto &l : layers) l.current_len = 0;
    }
    
    bool empty() const { return layers.empty(); }
};

/// Inference-only LLaMA forward pass — no autograd overhead.
/// Takes token IDs and produces logits using direct Tensor operations.
///
/// @param model  Loaded LLaMAModel (weights are read from its Nodes' .data)
/// @param token_ids  1D tensor of token IDs, shape [seq_len]
/// @return Logits tensor, shape [seq_len, vocab_size]
core::Tensor llama_forward_inference(const model::LLaMAModel &model,
                                     const core::Tensor &token_ids,
                                     KVCache *cache = nullptr);

/// Same but for a single token (optimized path for generation steps).
/// @param model  Loaded LLaMAModel
/// @param token_id  Single token ID
/// @param pos  Position offset for RoPE
/// @param cache  Optional KV-cache to update and use
/// @return Logits tensor, shape [1, vocab_size]
core::Tensor llama_forward_single_token(const model::LLaMAModel &model,
                                        int token_id, size_t pos,
                                        KVCache *cache = nullptr);

}  // namespace infer
}  // namespace serius
