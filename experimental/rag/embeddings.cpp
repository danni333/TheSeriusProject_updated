#include "embeddings.h"

#include <cmath>
#include <functional>
#include <numeric>

namespace serius {
namespace rag {

std::vector<float> compute_embedding(const std::string& text, size_t dimension)
{
    std::vector<float> embedding(dimension);

    // Simple hash-based embedding for testing
    // In production, this would use a dedicated embedding model
    std::hash<std::string> hasher;
    size_t hash = hasher(text);

    for (size_t i = 0; i < dimension; ++i) {
        // Generate pseudo-random values from hash
        size_t seed = hash + i * 31;
        seed = (seed ^ (seed >> 30)) * 0xbf58476d1cd4ab7;
        seed = (seed ^ (seed >> 27)) * 0x94d049bb133111eb;
        seed = seed ^ (seed >> 31);
        embedding[i] = static_cast<float>(seed) / static_cast<float>(std::numeric_limits<size_t>::max());
    }

    normalize_embedding(embedding);
    return embedding;
}

void normalize_embedding(std::vector<float>& embedding)
{
    float norm = l2_norm(embedding);
    if (norm > 1e-6f) {
        for (auto& val : embedding) {
            val /= norm;
        }
    }
}

float l2_norm(const std::vector<float>& vec)
{
    float sum_sq = 0.0f;
    for (float val : vec) {
        sum_sq += val * val;
    }
    return std::sqrt(sum_sq);
}

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size() || a.empty()) {
        return 0.0f;
    }

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);

    if (norm_a < 1e-6f || norm_b < 1e-6f) {
        return 0.0f;
    }

    return dot / (norm_a * norm_b);
}

}  // namespace rag
}  // namespace serius
