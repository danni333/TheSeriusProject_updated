#pragma once

#include <random>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace inference {

/// Configuration for sampling strategies.
struct SamplerConfig
{
    float temperature = 0.7f;      // Temperature scaling (0.0 = greedy)
    int top_k = 40;                // Top-K filtering (0 = disabled)
    float top_p = 0.9f;            // Top-P (nucleus) filtering (1.0 = disabled)
    float repetition_penalty = 1.0f;  // Repetition penalty (1.0 = disabled)
    unsigned seed = 42;            // Random seed for reproducibility
};

/// Sampler implements various token sampling strategies for text generation.
class Sampler
{
   public:
    explicit Sampler(const SamplerConfig &config = SamplerConfig());

    /// Sample a single token index from logits of shape (vocab_size,).
    /// Applies temperature, repetition penalty, top-k, top-p in sequence.
    int sample(const core::Tensor &logits, const std::vector<int> &past_tokens = {});

    /// Greedy decoding: return the argmax token.
    static int greedy(const core::Tensor &logits);

    /// Apply temperature scaling to logits (in-place).
    static void apply_temperature(core::Tensor &logits, float temperature);

    /// Apply repetition penalty to logits (in-place).
    static void apply_repetition_penalty(core::Tensor &logits, const std::vector<int> &past_tokens, float penalty);

    /// Apply top-k filtering: keep only the top-k logits, set rest to -inf.
    static void apply_top_k(core::Tensor &logits, int k);

    /// Apply top-p (nucleus) filtering: keep smallest set of tokens whose
    /// cumulative probability exceeds p, set rest to -inf.
    static void apply_top_p(core::Tensor &logits, float p);

    /// Compute softmax over logits and return probabilities.
    static core::Tensor softmax(const core::Tensor &logits);

    /// Categorical sampling from a probability distribution.
    /// Returns the sampled index.
    int categorical_sample(const core::Tensor &probs);

    // Accessors
    const SamplerConfig &config() const { return config_; }
    void set_config(const SamplerConfig &config) { config_ = config; }
    void set_seed(unsigned seed);

   private:
    SamplerConfig config_;
    std::mt19937 rng_;
};

}  // namespace inference
}  // namespace serius
