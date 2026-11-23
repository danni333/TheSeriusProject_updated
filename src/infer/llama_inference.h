#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "../tensor/tensor.h"
#include "../transformer/llama_model.h"

namespace serius {
namespace infer {

struct KVCacheLayer {
    KVCacheLayer() : k({0, 0}), v({0, 0}), current_len(0) {}
    core::Tensor k; // [max_seq_len, kv_dim]
    core::Tensor v; // [max_seq_len, kv_dim]
    // FP16 KV cache (CPU decode path): halves DRAM traffic vs fp32.
    std::vector<uint16_t> k16, v16;  // [max_seq_len, kv_dim]
    size_t current_len = 0;
};

struct KVCache {
    KVCache()
        : scratch_h({0, 0}), scratch_normed({0, 0}), scratch_q({0, 0}),
          scratch_k({0, 0}), scratch_v({0, 0}), scratch_attn_concat({0, 0}),
          scratch_mlp_gate({0, 0}), scratch_mlp_up({0, 0}), scratch_mlp_down({0, 0}),
          instance_id(next_instance_id()) {}

    /// Unique per-instance id: stack/heap addresses get reused across turns,
    /// so pointer comparison alone can't detect a destroyed+recreated cache
    /// (the CUDA decode graph embeds raw KV pointers — a stale graph writes
    /// into freed/reused GPU memory and produces garbage tokens).
    static uint64_t next_instance_id() {
        static std::atomic<uint64_t> counter{0};
        return ++counter;
    }
    uint64_t instance_id = 0;

    std::vector<KVCacheLayer> layers;

    // Single GPU arena backing all layers' K/V tensors on the CUDA path:
    // 112 separate cudaMallocs cost ~200 ms; one slab + views is ~1 call.
    core::Tensor gpu_arena;

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
/// @return Logits tensor, shape [seq_len, vocab_size].
///         Empty tensor on non-last pipeline-parallel ranks (they only
///         forward hidden states; logits exist solely on the last rank).
core::Tensor llama_forward_inference(const model::LLaMAModel &model,
                                     const core::Tensor &token_ids,
                                     KVCache *cache = nullptr);

/// Same as llama_forward_inference but returns logits for the last position
/// only, shape [1, vocab_size]. Cheaper on CUDA: skips the {seq_len, vocab}
/// logits tensor and its host copy.
core::Tensor llama_forward_inference_last(const model::LLaMAModel &model,
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

/// Offload a LLaMA model to the GPU: FP16 weights, transposed lm_head copies,
/// packed quantized weights upload, and cuBLAS/prefill warmup.
/// Returns true when the model was moved; false when CUDA is unavailable at
/// compile time or SERIUS_NO_GPU is set. Safe to call unconditionally.
bool offload_llama_model_to_gpu(model::LLaMAModel &model);

/// Whether a LLaMA model should be offloaded to the GPU.
enum class LlamaGpuFit {
    kFits,               ///< estimated footprint fits in free VRAM
    kNoCuda,             ///< CUDA unavailable (compile-time or runtime)
    kInsufficientVram,   ///< model needs more VRAM than is currently free
};

/// Plan for offloading `model` to the GPU: estimated footprint vs free VRAM.
struct LlamaGpuPlan {
    LlamaGpuFit fit = LlamaGpuFit::kNoCuda;
    size_t need_bytes = 0;   ///< estimated GPU memory the offload will use
    size_t free_bytes = 0;   ///< currently free VRAM (0 when unknown)
};

/// Estimate the GPU memory needed to offload this model (FP16 dense weights,
/// packed quantized weights, lm_head copies, KV cache at max context,
/// workspace margin) and compare it against currently free VRAM.
/// `need_bytes`/`free_bytes` are filled whenever they can be determined.
LlamaGpuPlan plan_llama_gpu_offload(const model::LLaMAModel &model);

/// Rough dense-equivalent parameter count from the model config — used for
/// logging/tiering ("~7B params"), not for exact memory accounting.
size_t estimate_llama_params(const model::LLaMAModel &model);

/// User-facing GPU preference, shared by `serius serve` and `serius chat`.
enum class LlamaGpuMode {
    kAuto,   ///< plan from model size vs free VRAM (default)
    kCpu,    ///< never touch the GPU
    kCuda,   ///< require the GPU; warn and stay on CPU only if it cannot fit
};

/// Outcome of resolve_llama_gpu: whether to offload plus a human-readable
/// reason (logged by callers so the per-model decision is visible).
struct LlamaGpuDecision {
    bool use_gpu = false;
    LlamaGpuFit fit = LlamaGpuFit::kNoCuda;
    size_t need_bytes = 0;
    size_t free_bytes = 0;
    size_t est_params = 0;
    const char *reason = "";
};

/// Decide GPU vs CPU for `model` under `mode`:
/// - kCpu   → never offload.
/// - kCuda  → offload when the plan says it fits (or CUDA is the only
///            option); stays on CPU with a warning when VRAM is insufficient.
/// - kAuto  → offload iff the estimated footprint fits in free VRAM.
LlamaGpuDecision resolve_llama_gpu(const model::LLaMAModel &model, LlamaGpuMode mode);

/// Partial GPU offload (--gpu-layers N, llama.cpp --n-gpu-layers style):
/// upload the first `n_layers` transformer blocks (+ embeddings) to the GPU
/// as dual-resident FP32 copies — host weights stay intact for the CPU
/// suffix. Decode runs a GPU prefix then a CPU suffix per token; prefill
/// stays on the per-token path so KV cache stays consistent.
/// Returns false when CUDA is unavailable or n_layers is out of range.
bool offload_llama_model_partial(model::LLaMAModel &model, size_t n_layers);

/// Number of leading layers offloaded by offload_llama_model_partial
/// (0 = no partial offload). Consulted by the inference dispatch.
size_t partial_gpu_layers(const model::LLaMAModel &model);

}  // namespace infer
}  // namespace serius
