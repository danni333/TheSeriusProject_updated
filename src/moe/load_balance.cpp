#include "load_balance.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace serius {
namespace model {

float load_balance_loss(const core::Tensor &router_probs,
                        const std::vector<std::vector<size_t>> &assignments,
                        size_t n_experts,
                        float alpha)
{
    if (n_experts == 0) {
        return 0.0f;
    }
    if (alpha < 0.0f) {
        throw std::invalid_argument("load_balance_loss alpha must be non-negative");
    }
    if (router_probs.ndim() != 2 || router_probs.shape()[1] < n_experts) {
        throw std::invalid_argument("load_balance_loss expects router probabilities with shape [tokens, experts]");
    }

    std::vector<float> load_fraction(n_experts, 0.0f);
    std::vector<float> router_fraction(n_experts, 0.0f);
    const size_t total_tokens = router_probs.shape()[0];
    if (total_tokens == 0) {
        return 0.0f;
    }
    if (!assignments.empty() && assignments.size() != total_tokens) {
        throw std::invalid_argument("load_balance_loss assignments must have one entry per token");
    }

    for (size_t token = 0; token < total_tokens; ++token) {
        size_t top_expert = 0;
        float top_probability = router_probs.at(token, 0);
        for (size_t expert = 0; expert < n_experts; ++expert) {
            const float probability = router_probs.at(token, expert);
            router_fraction[expert] += probability / static_cast<float>(total_tokens);
            if (probability > top_probability) {
                top_probability = probability;
                top_expert = expert;
            }
        }

        if (!assignments.empty()) {
            for (size_t expert : assignments[token]) {
                if (expert < n_experts) {
                    load_fraction[expert] += 1.0f / static_cast<float>(total_tokens);
                }
            }
        } else {
            load_fraction[top_expert] += 1.0f / static_cast<float>(total_tokens);
        }
    }

    float loss = 0.0f;
    for (size_t expert = 0; expert < n_experts; ++expert) {
        loss += load_fraction[expert] * router_fraction[expert];
    }
    return alpha * static_cast<float>(n_experts) * loss;
}

float load_balance_loss(const core::Tensor &router_probs,
                        const std::vector<std::vector<int>> &assignments,
                        size_t n_experts,
                        float alpha)
{
    std::vector<std::vector<size_t>> normalized(assignments.size());
    for (size_t token = 0; token < assignments.size(); ++token) {
        normalized[token].reserve(assignments[token].size());
        for (int expert : assignments[token]) {
            normalized[token].push_back(static_cast<size_t>(expert));
        }
    }
    return load_balance_loss(router_probs, normalized, n_experts, alpha);
}

}  // namespace model
}  // namespace serius
