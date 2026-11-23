// EXPERIMENTAL: modulo didattico/sperimentale — vedi docs/experimental.md

#pragma once

#include <map>
#include <mutex>
#include <utility>
#include <vector>

#include "infer/activation_hook.h"
#include "tensor/tensor.h"

namespace serius {
namespace model {
class LLaMAModel;
}
namespace interpret {

using core::Tensor;

/// Runs one real LLaMA forward pass with infer::ActivationHooks installed
/// and stores copies of the captured activations.
///
/// Scope (honest limits):
///  - CPU inference path only — CUDA kernels do not emit hooks.
///  - llama_forward_inference / _last / _single_token all funnel through
///    the same layer loop, so every token position is captured.
///  - Attention rows are captured per (layer, head, query row) in
///    chronological order; with KV-cache decode each call contributes one
///    row whose npos equals the number of attended positions.
///  - Hooks are process-global: do not run concurrent captures.
class InferenceCapture
{
   public:
    InferenceCapture() = default;
    ~InferenceCapture();
    InferenceCapture(const InferenceCapture &) = delete;
    InferenceCapture &operator=(const InferenceCapture &) = delete;

    /// Install hooks, run the forward pass over token_ids (shape [seq_len],
    /// float token ids as produced by BPETokenizer), then uninstall hooks.
    /// Uses a private KV cache so attention sees the full causal context.
    /// Returns the logits tensor produced by llama_forward_inference.
    Tensor run(const model::LLaMAModel &model, const Tensor &token_ids);

    /// Number of transformer layers observed.
    size_t num_layers() const;

    /// Hidden states captured for a layer: one Tensor {seq_len, hidden_dim}
    /// per forward call that touched the layer (decode => many {1,hidden}).
    const std::vector<Tensor> &hidden_calls(int layer) const;

    /// Last position of the LAST forward call for `layer`, shape {1,hidden}.
    /// This is the state LogitLens expects (it predicts the next token).
    Tensor hidden_last_position(int layer) const;

    /// Attention distribution rows for (layer, head), chronological order.
    /// Each Tensor is {npos}: probs[kj] for key position kj < npos.
    const std::vector<Tensor> &attention_rows(int layer, int head) const;

    /// Number of distinct heads observed in a layer (0 if none).
    size_t num_attention_heads(int layer) const;

    /// Clear all captured data (hooks are already removed after run()).
    void clear();

   private:
    friend void capture_layer_hidden(int, const float *, size_t, size_t);
    friend void capture_attention_row(int, int, size_t, const float *, size_t);

    void record_hidden(int layer, const float *data, size_t seq_len, size_t hidden_dim);
    void record_attention(int layer, int head, size_t query_idx,
                          const float *probs, size_t npos);

    void ensure_layer(int layer);

    mutable std::mutex mutex_;
    // hidden_[layer][call] -> {seq_len, hidden_dim} owning copy
    std::vector<std::vector<Tensor>> hidden_;
    // attn_[layer][head] -> rows in arrival order, each {npos}
    std::vector<std::vector<std::vector<Tensor>>> attn_;
};

/// LM head in the {vocab_size, hidden_dim} row-major layout that
/// LogitLens::compute expects. Returns model.lm_head_t() when the loader
/// built it (safetensors path), otherwise a transposed copy of
/// model.lm_head() which is stored {hidden_dim, vocab_size}.
/// Empty tensor when the model has no LM head.
Tensor logit_lens_head(const model::LLaMAModel &model);

}  // namespace interpret
}  // namespace serius
