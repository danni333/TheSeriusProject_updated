#include "ppo_trainer.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace serius {
namespace rlhf {

struct PPOTrainer::Impl
{
    model::LLamaConfig config;
    PPOTrainer::Config ppo_config;
    std::unique_ptr<RewardModel> reward_model;

    explicit Impl(const model::LLamaConfig &cfg, const PPOTrainer::Config &pcfg)
        : config(cfg), ppo_config(pcfg), reward_model(new RewardModel(cfg))
    {
    }
};

PPOTrainer::PPOTrainer(const model::LLamaConfig &model_config, const Config &ppo_config)
    : impl_(new Impl(model_config, ppo_config))
{
}

PPOTrainer::PPOTrainer(const model::LLamaConfig &model_config)
    : PPOTrainer(model_config, Config{})
{
}

PPOTrainer::~PPOTrainer() = default;

std::vector<float> PPOTrainer::compute_logprobs(const Tensor &tokens) const
{
    return impl_->reward_model->compute_logprobs(tokens);
}

RewardModel::GenerationResult PPOTrainer::generate_with_logprobs(const Tensor &prompt_tokens,
                                                                 int max_new_tokens) const
{
    return impl_->reward_model->generate_with_logprobs(prompt_tokens, max_new_tokens);
}

float PPOTrainer::train_step(const std::vector<Tensor> &prompts)
{
    if (prompts.empty()) return 0.0f;

    const size_t G = impl_->ppo_config.group_size;
    const float clip_eps = impl_->ppo_config.clip_eps;
    const float kl_beta = impl_->ppo_config.kl_beta;
    const float advantage_eps = impl_->ppo_config.advantage_eps;

    // Step 1: For each prompt, generate G responses and collect rewards
    std::vector<float> all_rewards;
    std::vector<std::vector<float>> all_log_probs_new;
    std::vector<std::vector<float>> all_log_probs_old;
    std::vector<Tensor> all_responses;

    for (const auto &prompt : prompts) {
        for (size_t g = 0; g < G; ++g) {
            auto gen = generate_with_logprobs(prompt, 16);
            float reward = impl_->reward_model->score(gen.tokens);

            all_rewards.push_back(reward);
            all_log_probs_new.push_back(gen.log_probs);
            all_log_probs_old.push_back(gen.log_probs);  // Same as old for first iteration
            all_responses.push_back(gen.tokens);
        }
    }

    // Step 2: Compute relative advantages (GRPO-style)
    std::vector<float> advantages = compute_advantages(all_rewards, advantage_eps);

    // Step 3: Compute PPO loss with KL penalty
    float total_loss = 0.0f;
    size_t num_tokens = 0;

    for (size_t i = 0; i < all_responses.size(); ++i) {
        // KL penalty: compute log_probs from reference model (same as old for now)
        float kl = kl_divergence(all_log_probs_new[i], all_log_probs_old[i]);
        float adjusted_reward = kl_penalty(all_rewards[i], kl, kl_beta);

        // Use adjusted reward as advantage (simplified: advantage is already relative)
        float advantage = advantages[i];

        // Compute PPO loss for this response
        float response_loss = ppo_loss(all_log_probs_new[i], all_log_probs_old[i], advantage, clip_eps);
        total_loss += response_loss;
        num_tokens += all_log_probs_new[i].size();
    }

    if (num_tokens > 0) {
        total_loss /= static_cast<float>(num_tokens);
    }

    return total_loss;
}

}  // namespace rlhf
}  // namespace serius