#include "logit_lens.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace serius {
namespace interpret {

std::vector<LogitLens::Prediction> LogitLens::compute(const Tensor &hidden, const Tensor &lm_head,
                                                      int top_k) const
{
    std::vector<Prediction> result;

    // hidden: (d_model,) or (1, d_model)
    // lm_head: (vocab_size, d_model)
    size_t d_model = hidden.shape().back();
    size_t vocab_size = lm_head.shape()[0];

    // Compute logits = lm_head @ hidden
    std::vector<float> logits(vocab_size, 0.0f);
    const float *h = hidden.data();
    const float *w = lm_head.data();

    for (size_t v = 0; v < vocab_size; ++v) {
        float dot = 0.0f;
        for (size_t d = 0; d < d_model; ++d) {
            dot += w[v * d_model + d] * h[d];
        }
        logits[v] = dot;
    }

    // Find top-k
    std::vector<std::pair<int, float>> indexed;
    indexed.reserve(vocab_size);
    for (size_t v = 0; v < vocab_size; ++v) {
        indexed.push_back({static_cast<int>(v), logits[v]});
    }

    std::partial_sort(indexed.begin(), indexed.begin() + std::min(static_cast<size_t>(top_k), vocab_size),
                      indexed.end(),
                      [](const auto &a, const auto &b) { return a.second > b.second; });

    result.reserve(top_k);
    for (int i = 0; i < std::min(static_cast<size_t>(top_k), vocab_size); ++i) {
        Prediction p;
        p.token_id = indexed[i].first;
        p.logit = indexed[i].second;
        p.token = token_text(p.token_id);
        result.push_back(p);
    }

    return result;
}

std::string LogitLens::token_text(int token_id) const
{
    if (token_id >= 0 && static_cast<size_t>(token_id) < vocab_.size()) {
        return vocab_[token_id];
    }
    return "[" + std::to_string(token_id) + "]";
}

}  // namespace interpret
}  // namespace serius