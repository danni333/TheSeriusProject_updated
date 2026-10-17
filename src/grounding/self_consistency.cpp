#include "self_consistency.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace serius {
namespace grounding {

float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.size() != b.size() || a.empty()) {
        return 0.0f;
    }
    float dot = 0.0f;
    float na = 0.0f;
    float nb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na == 0.0f || nb == 0.0f) {
        return 0.0f;
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

size_t self_consistent_index(const std::vector<std::vector<float>> &embeddings)
{
    if (embeddings.empty()) {
        return 0;
    }
    size_t best = 0;
    float best_sim = -1.0f;
    for (size_t i = 0; i < embeddings.size(); ++i) {
        float sum = 0.0f;
        size_t count = 0;
        for (size_t j = 0; j < embeddings.size(); ++j) {
            if (i == j) {
                continue;
            }
            sum += cosine_similarity(embeddings[i], embeddings[j]);
            ++count;
        }
        const float avg = count > 0 ? sum / static_cast<float>(count) : 0.0f;
        if (avg > best_sim) {
            best_sim = avg;
            best = i;
        }
    }
    return best;
}

std::string majority_vote(const std::vector<std::string> &responses)
{
    std::map<std::string, int> counts;
    for (const auto &r : responses) {
        counts[r]++;
    }
    std::string best;
    int best_count = -1;
    for (const auto &kv : counts) {
        if (kv.second > best_count) {
            best_count = kv.second;
            best = kv.first;
        }
    }
    return best;
}

}  // namespace grounding
}  // namespace serius
