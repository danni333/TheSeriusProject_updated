#include "ppo.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace serius {
namespace rlhf {

/// KL divergence penalty between two policies
/// D_KL(pi_new || pi_ref) = sum(pi_new * log(pi_new / pi_ref))
float kl_divergence(const std::vector<float> &log_probs_new,
                    const std::vector<float> &log_probs_ref)
{
    float kl = 0.0f;
    const size_t n = std::min(log_probs_new.size(), log_probs_ref.size());
    for (size_t i = 0; i < n; ++i) {
        const float p = std::exp(log_probs_new[i]);
        kl += p * (log_probs_new[i] - log_probs_ref[i]);
    }
    return kl;
}

/// KL penalty applied to reward: R_final = R_reward - beta * KL
float kl_penalty(float reward_score, float kl, float beta)
{
    return reward_score - beta * kl;
}

/// Compute mean of a vector
float mean(const std::vector<float> &values)
{
    if (values.empty()) {
        return 0.0f;
    }
    return std::accumulate(values.begin(), values.end(), 0.0f) /
           static_cast<float>(values.size());
}

/// Compute standard deviation of a vector
float std_dev(const std::vector<float> &values)
{
    if (values.empty()) {
        return 0.0f;
    }
    const float m = mean(values);
    float var = 0.0f;
    for (float v : values) {
        const float diff = v - m;
        var += diff * diff;
    }
    var /= static_cast<float>(values.size());
    return std::sqrt(var);
}

/// Compute relative advantage: A_i = (R_i - mean(R)) / (std(R) + eps)
std::vector<float> compute_advantages(const std::vector<float> &rewards, float eps)
{
    std::vector<float> advantages;
    if (rewards.empty()) {
        return advantages;
    }
    advantages.reserve(rewards.size());
    const float m = mean(rewards);
    const float s = std_dev(rewards);
    for (float r : rewards) {
        advantages.push_back((r - m) / (s + eps));
    }
    return advantages;
}

/// Clipped PPO objective
/// ratio = pi_new / pi_old
/// advantage = computed advantage
/// clip_eps = clipping parameter (default 0.2)
float ppo_clipped_objective(float ratio, float advantage, float clip_eps)
{
    const float clipped = std::clamp(ratio, 1.0f - clip_eps, 1.0f + clip_eps);
    return std::fmin(ratio * advantage, clipped * advantage);
}

/// PPO loss for a single token
float ppo_loss(const std::vector<float> &log_probs_new,
               const std::vector<float> &log_probs_old,
               float advantage, float clip_eps)
{
    const size_t n = std::min(log_probs_new.size(), log_probs_old.size());
    if (n == 0) {
        return 0.0f;
    }
    float total = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float ratio = std::exp(log_probs_new[i] - log_probs_old[i]);
        total += ppo_clipped_objective(ratio, advantage, clip_eps);
    }
    return -total / static_cast<float>(n);
}

}  // namespace rlhf
}  // namespace serius
