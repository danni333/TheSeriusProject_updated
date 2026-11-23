#pragma once

#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace rlhf {

using core::Tensor;

/// KL divergence penalty between two policies
/// D_KL(pi_new || pi_ref) = sum(pi_new * log(pi_new / pi_ref))
float kl_divergence(const std::vector<float> &log_probs_new,
                    const std::vector<float> &log_probs_ref);

/// KL penalty applied to reward: R_final = R_reward - beta * KL
float kl_penalty(float reward_score, float kl, float beta = 0.1f);

/// Compute mean of a vector
float mean(const std::vector<float> &values);

/// Compute standard deviation of a vector
float std_dev(const std::vector<float> &values);

/// Compute relative advantage: A_i = (R_i - mean(R)) / (std(R) + eps)
std::vector<float> compute_advantages(const std::vector<float> &rewards, float eps = 1e-8f);

/// Clipped PPO objective
/// ratio = pi_new / pi_old
/// advantage = computed advantage
/// clip_eps = clipping parameter (default 0.2)
float ppo_clipped_objective(float ratio, float advantage, float clip_eps = 0.2f);

/// PPO loss for a single token
float ppo_loss(const std::vector<float> &log_probs_new,
               const std::vector<float> &log_probs_old,
               float advantage, float clip_eps = 0.2f);

}  // namespace rlhf
}  // namespace serius