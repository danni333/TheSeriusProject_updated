#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace serius {
namespace infer {

Sampler::Sampler(float temperature, int top_k, float top_p, uint64_t seed)
    : temperature_(temperature), top_k_(top_k), top_p_(top_p), rng_(static_cast<std::mt19937::result_type>(seed))
{
}

int Sampler::sample(const core::Tensor &logits) const
{
    return sample(logits, {}, 1.0f);
}

int Sampler::sample(const core::Tensor &logits, const std::vector<int> &recent_tokens, float repetition_penalty) const
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

    const bool need_penalty = repetition_penalty > 1.0f && !recent_tokens.empty();
    thread_local std::vector<float> work_storage;
    const float *work = ptr;
    if (need_penalty) {
        work_storage.assign(ptr, ptr + vocab_size);
        for (int tid : recent_tokens) {
            if (tid >= 0 && static_cast<size_t>(tid) < vocab_size) {
                if (work_storage[tid] > 0.0f) {
                    work_storage[tid] /= repetition_penalty;
                } else {
                    work_storage[tid] *= repetition_penalty;
                }
            }
        }
        work = work_storage.data();
    }

    if (temperature_ <= 0.0f) {
        return static_cast<int>(std::max_element(work, work + vocab_size) - work);
    }

    int k = (top_k_ > 0) ? static_cast<int>(std::min(static_cast<size_t>(top_k_), vocab_size))
                         : static_cast<int>(vocab_size);

    // Pack (order-preserving logit key << 32) | index into uint64: integer
    // nth_element over 8-byte keys is ~2-3x faster than pair<float,size_t>.
    // key = float_bits ^ (sign ? ~0 : 0x80000000) preserves float ordering.
    auto fkey = [](float f) -> uint32_t {
        uint32_t b;
        std::memcpy(&b, &f, sizeof(b));
        return b ^ ((b & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
    };
    auto keyf = [](uint32_t k) -> float {
        const uint32_t b = k ^ ((k & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu);
        float f;
        std::memcpy(&f, &b, sizeof(f));
        return f;
    };

    thread_local std::vector<uint64_t> keys;
    keys.resize(vocab_size);  // persistent: avoids ~400KB of page faults per token
    const float inv_t = 1.0f / temperature_;
    for (size_t i = 0; i < vocab_size; ++i) {
        keys[i] = (static_cast<uint64_t>(fkey(work[i] * inv_t)) << 32) | i;
    }

    auto cmp_desc = [](uint64_t a, uint64_t b) { return a > b; };

    if (static_cast<size_t>(k) < keys.size()) {
        std::nth_element(keys.begin(), keys.begin() + k, keys.end(), cmp_desc);
        std::sort(keys.begin(), keys.begin() + k, cmp_desc);
        keys.resize(static_cast<size_t>(k));
    } else {
        std::sort(keys.begin(), keys.end(), cmp_desc);
    }

    const size_t nk = keys.size();
    std::vector<float> probs(nk);
    const float max_val = nk > 0 ? keyf(static_cast<uint32_t>(keys[0] >> 32)) : 0.0f;
    for (size_t i = 0; i < nk; ++i) {
        probs[i] = std::exp(keyf(static_cast<uint32_t>(keys[i] >> 32)) - max_val);
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
    return static_cast<int>(keys[chosen] & 0xFFFFFFFFull);
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
