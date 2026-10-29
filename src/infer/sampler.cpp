#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace serius {
namespace infer {

Sampler::Sampler(float temperature, int top_k, float top_p, uint64_t seed)
    : temperature_(temperature), top_k_(top_k), top_p_(top_p), rng_(static_cast<std::mt19937::result_type>(seed))
{
}

int Sampler::sample(const core::Tensor &logits) const
{
    size_t vocab_size = logits.shape().back();
    const float *ptr = logits.data();
    if (ptr == nullptr) {
        return 0;
    }
    if (logits.ndim() == 2) {
        size_t seq_len = logits.shape()[0];
        ptr += (seq_len - 1) * vocab_size;
    }

    std::vector<std::pair<float, size_t>> indexed;
    indexed.reserve(vocab_size);
    float inv_t = (temperature_ > 0.f) ? (1.f / temperature_) : 1e6f;
    for (size_t i = 0; i < vocab_size; ++i) {
        indexed.emplace_back(ptr[i] * inv_t, i);
    }
    std::sort(indexed.begin(), indexed.end(),
              [](const std::pair<float, size_t> &a, const std::pair<float, size_t> &b) { return a.first > b.first; });

    int k = (top_k_ > 0) ? static_cast<int>(std::min(static_cast<size_t>(top_k_), vocab_size))
                         : static_cast<int>(vocab_size);
    std::vector<float> probs(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
        probs[static_cast<size_t>(i)] = std::exp(indexed[static_cast<size_t>(i)].first);
    }
    float sum = std::accumulate(probs.begin(), probs.end(), 0.f);
    if (sum > 0.f) {
        for (float &p : probs) {
            p /= sum;
        }
    }
    if (top_p_ > 0.f && top_p_ < 1.f) {
        float cum = 0.f;
        size_t n = 0;
        for (size_t i = 0; i < probs.size(); ++i) {
            cum += probs[i];
            n = i + 1;
            if (cum >= top_p_) {
                break;
            }
        }
        probs.resize(n);
        sum = std::accumulate(probs.begin(), probs.end(), 0.f);
        if (sum > 0.f) {
            for (float &p : probs) {
                p /= sum;
            }
        }
    }
    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
    size_t chosen = dist(rng_);
    return static_cast<int>(indexed[chosen].second);
}

int Sampler::sample_greedy(const core::Tensor &logits) const
{
    size_t vocab_size = logits.shape().back();
    const float *ptr = logits.data();
    if (ptr == nullptr) {
        return 0;
    }
    if (logits.ndim() == 2) {
        size_t seq_len = logits.shape()[0];
        ptr += (seq_len - 1) * vocab_size;
    }
    return static_cast<int>(std::max_element(ptr, ptr + vocab_size) - ptr);
}

}  // namespace infer
}  // namespace serius
