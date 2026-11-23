#pragma once

#include <cstddef>

namespace serius {
namespace infer {

/// Opt-in observability hooks for interpretability tooling
/// (src/interpret). Hooks are PROCESS-GLOBAL (not thread-local) because
/// attention rows are produced inside OpenMP worker threads. Callbacks
/// may therefore run concurrently — consumers must synchronize.
/// Pointed-to data is only valid for the duration of the call — copy it
/// if you need to keep it.
///
/// Supported on the CPU path and the eager CUDA paths (batched prefill and
/// single-token decode). The CUDA-graph decode path cannot emit hooks —
/// llama_forward_single_token_cuda_graph() automatically falls back to the
/// eager path while hooks are installed. In the fused-GQA CUDA decode branch
/// (FP16 + KV cache <= 4096) attention probabilities are never materialized,
/// so attention_row does not fire there; layer_hidden still does.
/// When no hooks are set the overhead is a single pointer check per
/// layer/head-row.
struct ActivationHooks
{
    /// After each transformer layer (post attention+MLP residual):
    ///   hidden[layer][s][d], row-major [seq_len, hidden_dim].
    /// For single-token decode seq_len == 1.
    void (*layer_hidden)(int layer_idx, const float *hidden, size_t seq_len, size_t hidden_dim) = nullptr;

    /// Attention distribution for one query row of one head, right after
    /// the causal softmax:
    ///   probs[kj] = attention weight on key position kj, kj in [0, npos).
    /// Entries beyond npos are masked (~0). query_idx is the row index
    /// within the current forward call (0 for single-token decode).
    void (*attention_row)(int layer_idx, int head_idx, size_t query_idx, const float *probs, size_t npos) = nullptr;
};

/// Install hooks process-wide. Pass {} (or clear) to disable.
/// Not safe to toggle concurrently with in-flight inference.
void set_activation_hooks(const ActivationHooks &hooks);

/// Remove hooks process-wide.
void clear_activation_hooks();

/// Whether any hook is currently installed on this thread.
bool activation_hooks_enabled();

/// Current hooks (all-null when unset). Cheap to call; check the fields
/// you need rather than enabled() when hoisting out of hot loops.
const ActivationHooks &activation_hooks();

}  // namespace infer
}  // namespace serius
