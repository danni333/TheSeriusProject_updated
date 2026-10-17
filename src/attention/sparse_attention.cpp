#include "sparse_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace serius {
namespace attention {

float cosine_similarity(const float* a, const float* b, size_t dim)
{
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < dim; ++i) {
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

float scaled_dot_product(const float* a, const float* b, size_t dim, float scale)
{
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return dot * scale;
}

SparseAttentionResult dynamic_sparse_attention(
    const Tensor& Q, const Tensor& K, const Tensor& V,
    float threshold, size_t top_k)
{
    const size_t seq_len = Q.shape()[0];
    const size_t d_model = Q.shape()[1];

    SparseAttentionResult result;
    result.output = Tensor({seq_len, d_model});
    result.output.fill(0.0f);
    result.active_keys.resize(seq_len);

    const float scale = 1.0f / std::sqrt(static_cast<float>(d_model));
    size_t total_pairs = 0;
    size_t active_pairs = 0;

    for (size_t i = 0; i < seq_len; ++i) {
        const float* q_row = Q.data() + i * d_model;

        // Calculate scores for all causal keys (j <= i)
        std::vector<std::pair<size_t, float>> scores;
        for (size_t j = 0; j <= i; ++j) {
            const float* k_row = K.data() + j * d_model;
            float score = scaled_dot_product(q_row, k_row, d_model, scale);
            scores.emplace_back(j, score);
            total_pairs++;
        }

        // Select active keys based on threshold or top-k
        if (top_k > 0) {
            // Select top-k
            std::partial_sort(scores.begin(), scores.begin() + std::min(top_k, scores.size()),
                             scores.end(),
                             [](const auto& a, const auto& b) { return a.second > b.second; });
            scores.resize(std::min(top_k, scores.size()));
        } else {
            // Select by threshold
            auto it = std::remove_if(scores.begin(), scores.end(),
                                     [threshold](const auto& p) { return p.second < threshold; });
            scores.erase(it, scores.end());
        }

        active_pairs += scores.size();

        // Store active key indices
        for (const auto& [key_idx, score] : scores) {
            result.active_keys[i].push_back(static_cast<int>(key_idx));
        }

        // If no active keys, use all keys (fallback to dense)
        if (scores.empty()) {
            for (size_t j = 0; j <= i; ++j) {
                result.active_keys[i].push_back(static_cast<int>(j));
                scores.emplace_back(j, scaled_dot_product(q_row, K.data() + j * d_model, d_model, scale));
            }
            active_pairs += i + 1;
        }

        // Apply softmax to scores
        float max_score = -std::numeric_limits<float>::infinity();
        for (const auto& [_, score] : scores) {
            max_score = std::max(max_score, score);
        }

        float sum_exp = 0.0f;
        std::vector<float> exp_scores(scores.size());
        for (size_t k = 0; k < scores.size(); ++k) {
            exp_scores[k] = std::exp(scores[k].second - max_score);
            sum_exp += exp_scores[k];
        }

        // Weighted sum of values
        float* output_row = result.output.data() + i * d_model;
        for (size_t k = 0; k < scores.size(); ++k) {
            size_t key_idx = scores[k].first;
            float weight = exp_scores[k] / sum_exp;
            const float* v_row = V.data() + key_idx * d_model;

            for (size_t d = 0; d < d_model; ++d) {
                output_row[d] += weight * v_row[d];
            }
        }
    }

    result.sparsity_ratio = 1.0f - static_cast<float>(active_pairs) / static_cast<float>(total_pairs);

    return result;
}

}  // namespace attention
}  // namespace serius
