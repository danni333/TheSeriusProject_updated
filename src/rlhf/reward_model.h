#pragma once

#include <memory>
#include <vector>

#include "../autograd/node.h"
#include "../tensor/tensor.h"
#include "../transformer/llama_model.h"

namespace serius {
namespace rlhf {

using core::Tensor;
using core::autograd::Node;

/// Reward Model: LLaMA Transformer with a scalar output head.
/// Takes tokenized sequences and returns a scalar reward score.
class RewardModel
{
   public:
    explicit RewardModel(const model::LLamaConfig &config);
    RewardModel();
    ~RewardModel();

    RewardModel(const RewardModel &) = delete;
    RewardModel &operator=(const RewardModel &) = delete;

    /// Score a sequence of tokens
    /// tokens: 1D tensor of integer token ids, shape (seq_len,)
    /// returns scalar reward score
    float score(const Tensor &tokens) const;

    /// Compute log probabilities for a sequence (used in KL penalty)
    std::vector<float> compute_logprobs(const Tensor &tokens) const;

    /// Generate tokens with log probabilities
    struct GenerationResult
    {
        Tensor tokens;
        std::vector<float> log_probs;
    };
    GenerationResult generate_with_logprobs(const Tensor &prompt_tokens, int max_new_tokens = 16) const;

    const model::LLamaConfig &config() const { return config_; }

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    model::LLamaConfig config_;
};

/// Bradley-Terry preference loss
/// L = -log(sigmoid(score_win - score_lose))
float reward_preference_loss(float score_win, float score_lose);

}  // namespace rlhf
}  // namespace serius