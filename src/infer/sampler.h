#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "../common/defaults.h"
#include "../tensor/tensor.h"

namespace serius {
namespace infer {

/// Sampler for next-token generation from logits.
/// Supports temperature, top_k, top_p (nucleus).
class Sampler
{
   public:
    Sampler(float temperature = defaults::kTemperature, int top_k = defaults::kTopK, float top_p = defaults::kTopP,
            uint64_t seed = 0);

    /// Sample one token from logits (shape: (vocab_size,) or last row of (seq, vocab_size)).
    /// Uses temperature scaling, then optional top_k, top_p, and repetition penalty.
    int sample(const core::Tensor &logits) const;
    int sample(const core::Tensor &logits, const std::vector<int> &recent_tokens,
               float repetition_penalty = 1.0f) const;

    /// Greedy decode: argmax.
    int sample_greedy(const core::Tensor &logits) const;

    /// Sample from a prefiltered candidate set (e.g. GPU top-k over the raw
    /// logits). `cand_logits` are the RAW logits for `cand_idx`; the repetition
    /// penalty is applied here per candidate, exactly as in sample(). The
    /// candidate set must be a superset of the penalized top_k (a margin of
    /// ~top_k + top_k is sufficient since penalties only lower logits).
    /// Returns -1 if the candidate set is empty.
    int sample_from_candidates(const std::vector<int> &cand_idx, const std::vector<float> &cand_logits,
                               const std::vector<int> &recent_tokens, float repetition_penalty) const;

    /// Same as above but takes a dense occurrence-count array (indexed by
    /// token id, size > max id) instead of the recent_tokens list — avoids
    /// rebuilding a hash map on every decode step.
    int sample_from_candidates_dense(const std::vector<int> &cand_idx, const std::vector<float> &cand_logits,
                                     const int *token_counts, size_t counts_size, float repetition_penalty) const;

    void set_temperature(float t) { temperature_ = t; }
    void set_top_k(int k) { top_k_ = k; }
    void set_top_p(float p) { top_p_ = p; }
    void set_seed(uint64_t seed) { rng_.seed(static_cast<std::mt19937::result_type>(seed)); }

   private:
    int sample_from_penalized(const std::vector<int> &cand_idx, const std::vector<float> &penalized) const;

    float temperature_;
    int top_k_;
    float top_p_;
    mutable std::mt19937 rng_;
};

/// GPU-accelerated sampling: top-k preselection on device + CPU candidate
/// sampling, avoiding the full-vocab scan on the host. `logits` must be the
/// raw logits tensor (row `row` is used when 2-D). Returns false when the
/// tensor isn't GPU-resident or CUDA is unavailable — callers should fall
/// back to Sampler::sample(). No repetition penalty is applied.
bool sample_gpu_candidates(const core::Tensor &logits, size_t row, const Sampler &sampler, int &out_id);

}  // namespace infer
}  // namespace serius
