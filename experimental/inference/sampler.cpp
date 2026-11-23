#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace serius {
namespace inference {

Sampler::Sampler(const SamplerConfig &config) : config_(config), rng_(config.seed) {}

void Sampler::set_seed(unsigned seed)
{
    config_.seed = seed;
    rng_.seed(seed);
}

int Sampler::sample(const core::Tensor &logits, const std::vector<int> &past_tokens) const
{
    if (logits.ndim() != 1) {
        throw std::invalid_argument("Sampler::sample: logits must be 1D (vocab_size,)");
    }

    // Greedy shortcut
    if (config_.temperature <= 0.0f) {
        return greedy(logits);
    }

    // Work on a copy
    core::Tensor work = logits.copy();

    // 1. Repetition penalty
    if (config_.repetition_penalty != 1.0f && !past_tokens.empty()) {
        apply_repetition_penalty(work, past_tokens, config_.repetition_penalty);
    }

    // 2. Temperature scaling
    if (config_.temperature != 1.0f) {
        apply_temperature(work, config_.temperature);
    }

    // 3. Top-K filtering
    if (config_.top_k > 0 && config_.top_k < static_cast<int>(work.size())) {
        apply_top_k(work, config_.top_k);
    }

    // 4. Top-P (nucleus) filtering
    if (config_.top_p > 0.0f && config_.top_p < 1.0f) {
        apply_top_p(work, config_.top_p);
    }

    // 5. Softmax -> probabilities
    core::Tensor probs = softmax(work);

    // 6. Categorical sample
    return categorical_sample(probs);
}

int Sampler::greedy(const core::Tensor &logits)
{
    const float *data = logits.data();
    size_t n = logits.size();
    int best = 0;
    float best_val = data[0];
    for (size_t i = 1; i < n; i++) {
        if (data[i] > best_val) {
            best_val = data[i];
            best = static_cast<int>(i);
        }
    }
    return best;
}

void Sampler::apply_temperature(core::Tensor &logits, float temperature)
{
    float *data = logits.data();
    size_t n = logits.size();
    float inv_temp = 1.0f / temperature;
    for (size_t i = 0; i < n; i++) {
        data[i] *= inv_temp;
    }
}

void Sampler::apply_repetition_penalty(core::Tensor &logits, const std::vector<int> &past_tokens, float penalty)
{
    float *data = logits.data();
    size_t vocab_size = logits.size();
    for (int token : past_tokens) {
        if (token < 0 || static_cast<size_t>(token) >= vocab_size) continue;
        if (data[token] > 0.0f) {
            data[token] /= penalty;
        } else {
            data[token] *= penalty;
        }
    }
}

void Sampler::apply_top_k(core::Tensor &logits, int k)
{
    float *data = logits.data();
    size_t n = logits.size();

    // Sort indices by logit value descending
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [data](int a, int b) { return data[a] > data[b]; });

    // Mark which indices to keep
    std::vector<bool> keep(n, false);
    for (int i = 0; i < k; i++) {
        keep[indices[i]] = true;
    }

    // Mask everything not in top-k
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; i++) {
        if (!keep[i]) {
            data[i] = neg_inf;
        }
    }
}

void Sampler::apply_top_p(core::Tensor &logits, float p)
{
    size_t n = logits.size();

    // Compute softmax first to get probabilities for sorting
    core::Tensor probs = softmax(logits);
    const float *prob_data = probs.data();

    // Create index array sorted by probability descending
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [prob_data](int a, int b) { return prob_data[a] > prob_data[b]; });

    // Find cutoff where cumulative probability exceeds p
    float cumsum = 0.0f;
    size_t cutoff = n;
    for (size_t i = 0; i < n; i++) {
        cumsum += prob_data[indices[i]];
        if (cumsum >= p) {
            cutoff = i + 1;
            break;
        }
    }

    // Mask tokens beyond the cutoff
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    float *logits_data = logits.data();
    // Create a set of tokens to keep
    std::vector<bool> keep(n, false);
    for (size_t i = 0; i < cutoff; i++) {
        keep[indices[i]] = true;
    }
    for (size_t i = 0; i < n; i++) {
        if (!keep[i]) {
            logits_data[i] = neg_inf;
        }
    }
}

core::Tensor Sampler::softmax(const core::Tensor &logits)
{
    const float *data = logits.data();
    size_t n = logits.size();

    // Find max for numerical stability
    float max_val = data[0];
    for (size_t i = 1; i < n; i++) {
        if (data[i] > max_val) max_val = data[i];
    }

    // Compute exp and sum
    core::Tensor result({n});
    float *out = result.data();
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        out[i] = std::exp(data[i] - max_val);
        sum += out[i];
    }

    // Normalize
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < n; i++) {
            out[i] *= inv_sum;
        }
    }

    return result;
}

int Sampler::categorical_sample(const core::Tensor &probs) const
{
    const float *data = probs.data();
    size_t n = probs.size();

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);

    float cumsum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        cumsum += data[i];
        if (r <= cumsum) {
            return static_cast<int>(i);
        }
    }

    // Fallback: return last token (rounding error)
    return static_cast<int>(n - 1);
}

}  // namespace inference
}  // namespace serius
