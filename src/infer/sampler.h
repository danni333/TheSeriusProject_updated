#pragma once

#include "../tensor/tensor.h"

#include <cstdint>
#include <random>
#include <vector>

namespace serius {
namespace infer {

/// Sampler for next-token generation from logits.
/// Supports temperature, top_k, top_p (nucleus).
class Sampler
{
public:
    Sampler(float temperature = 0.7f, int top_k = 40, float top_p = 0.9f, uint64_t seed = 0);

    /// Sample one token from logits (shape: (vocab_size,) or last row of (seq, vocab_size)).
    /// Uses temperature scaling, then optional top_k and top_p.
    int sample(const core::Tensor &logits) const;

    /// Greedy decode: argmax.
    int sample_greedy(const core::Tensor &logits) const;

    void set_temperature(float t) { temperature_ = t; }
    void set_top_k(int k) { top_k_ = k; }
    void set_top_p(float p) { top_p_ = p; }
    void set_seed(uint64_t seed) { rng_.seed(static_cast<std::mt19937::result_type>(seed)); }

private:
    float temperature_;
    int top_k_;
    float top_p_;
    mutable std::mt19937 rng_;
};

}  // namespace infer
}  // namespace serius
