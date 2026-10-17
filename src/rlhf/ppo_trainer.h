#pragma once

#include <memory>
#include <vector>

#include "../tensor/tensor.h"
#include "../transformer/llama_model.h"
#include "ppo.h"
#include "reward_model.h"

namespace serius {
namespace rlhf {

using core::Tensor;

/// PPO Trainer: orchestrates policy optimization with reward model and KL penalty.
/// Supports GRPO-style advantage computation (no value model required).
class PPOTrainer
{
   public:
    struct Config
    {
        float clip_eps = 0.2f;
        float kl_beta = 0.1f;
        float lr = 1e-5f;
        size_t group_size = 4;  // G in GRPO
        float advantage_eps = 1e-8f;
    };

    PPOTrainer(const model::LLamaConfig &model_config, const Config &ppo_config = {});
    ~PPOTrainer();

    /// Single training step on a batch of prompts
    /// prompts: list of tokenized prompts (each is Tensor of shape (prompt_len,))
    /// returns the PPO loss value
    float train_step(const std::vector<Tensor> &prompts);

    /// Compute log probabilities for a sequence of tokens given the current policy
    std::vector<float> compute_logprobs(const Tensor &tokens) const;

    /// Generate a sequence of tokens with log probabilities
    struct GenerationResult
    {
        Tensor tokens;
        std::vector<float> log_probs;
    };
    GenerationResult generate_with_logprobs(const Tensor &prompt_tokens, int max_new_tokens = 16) const;

    const RewardModel &reward_model() const { return *impl_->reward_model; }
    const Config &config() const { return impl_->ppo_config; }

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rlhf
}  // namespace serius