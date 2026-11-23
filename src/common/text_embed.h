#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace serius {
namespace common {

/// Text embedder signature shared by grounding/safety/reasoning scorers.
/// Compatible with `rag::EmbedFn`; any callable with this signature works,
/// e.g. a lambda wrapping `hub::NomicBertEmbedder::embed`,
/// `hub::EmbeddingManager::compute_query_embedding`, or a remote
/// `/v1/embeddings` client.
using TextEmbedFn = std::function<std::vector<float>(const std::string &text)>;

inline float clamp01(float v)
{
    return std::max(0.0f, std::min(1.0f, v));
}

/// Cosine similarity between two vectors; 0 for empty/mismatched inputs.
inline float cosine_sim(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.empty() || a.size() != b.size()) {
        return 0.0f;
    }
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) {
        return 0.0f;
    }
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

/// Map a cosine similarity from the expected "related" range [lo, hi] to
/// [0, 1]. Text-embedding models typically produce cosines around 0.2-0.4 for
/// unrelated texts and 0.6-0.9 for related ones, so a raw clamp01(cosine)
/// inflates unrelated pairs; rescaling with a floor/ceiling fixes that.
/// Values outside [lo, hi] are clamped. Requires lo < hi.
inline float cosine_unit(float cosine, float lo, float hi)
{
    if (hi <= lo) {
        return clamp01(cosine);
    }
    return clamp01((cosine - lo) / (hi - lo));
}

}  // namespace common
}  // namespace serius
