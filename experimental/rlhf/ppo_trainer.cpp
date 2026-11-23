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

const RewardModel &PPOTrainer::reward_model() const
{
    return *impl_->reward_model;
}

const PPOTrainer::Config &PPOTrainer::config() const
{
    return impl_->ppo_config;
}

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

    // Step 2: GRPO-style advantages — KL-penalized rewards normalized
    // within each prompt's group of G samples (group-relative, not global).
    std::vector<float> adjusted_rewards(all_rewards.size());
    for (size_t i = 0; i < all_rewards.size(); ++i) {
        const float kl = kl_divergence(all_log_probs_new[i], all_log_probs_old[i]);
        adjusted_rewards[i] = kl_penalty(all_rewards[i], kl, kl_beta);
    }

    std::vector<float> advantages(all_rewards.size(), 0.0f);
    for (size_t p = 0; p < prompts.size(); ++p) {
        std::vector<float> group;
        group.reserve(G);
        for (size_t g = 0; g < G; ++g) {
            group.push_back(adjusted_rewards[p * G + g]);
        }
        const std::vector<float> group_adv =
            compute_advantages(group, advantage_eps);
        for (size_t g = 0; g < G && g < group_adv.size(); ++g) {
            advantages[p * G + g] = group_adv[g];
        }
    }

    // Step 3: Compute PPO loss with clipped objective
    float total_loss = 0.0f;
    size_t num_tokens = 0;

    for (size_t i = 0; i < all_responses.size(); ++i) {
        const float advantage = advantages[i];

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