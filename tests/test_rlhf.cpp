#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "rlhf/ppo.h"
#include "rlhf/reward_model.h"
#include "tensor/tensor.h"

using serius::core::Tensor;
using serius::model::LLamaConfig;
using serius::rlhf::RewardModel;
using serius::rlhf::compute_advantages;
using serius::rlhf::kl_divergence;
using serius::rlhf::kl_penalty;
using serius::rlhf::mean;
using serius::rlhf::ppo_clipped_objective;
using serius::rlhf::ppo_loss;
using serius::rlhf::reward_preference_loss;
using serius::rlhf::std_dev;

namespace {

LLamaConfig tiny_config()
{
    LLamaConfig c;
    c.vocab_size = 64;
    c.hidden_size = 16;
    c.num_attention_heads = 4;
    c.num_key_value_heads = 4;
    c.num_hidden_layers = 2;
    c.intermediate_size = 64;
    c.max_position_embeddings = 64;
    c.rms_norm_eps = 1e-5f;
    c.rope_theta = 10000.0f;
    return c;
}

}  // namespace

TEST(RewardModel, ScoreIsFiniteAndDeterministic)
{
    RewardModel rm(tiny_config());
    Tensor tokens({5});
    tokens[0] = 1.0f;
    tokens[1] = 2.0f;
    tokens[2] = 3.0f;
    tokens[3] = 4.0f;
    tokens[4] = 5.0f;

    const float s1 = rm.score(tokens);
    const float s2 = rm.score(tokens);

    EXPECT_TRUE(std::isfinite(s1));
    EXPECT_EQ(s1, s2);  // deterministic (seeded rand)
}

TEST(RewardModel, ComputeLogprobsShapeAndRange)
{
    RewardModel rm(tiny_config());
    Tensor tokens({4});
    tokens[0] = 1.0f;
    tokens[1] = 2.0f;
    tokens[2] = 3.0f;
    tokens[3] = 4.0f;

    const std::vector<float> log_probs = rm.compute_logprobs(tokens);
    EXPECT_EQ(log_probs.size(), 4u);
    for (float lp : log_probs) {
        EXPECT_TRUE(std::isfinite(lp));
        EXPECT_LE(lp, 0.0f);  // log-probabilities are <= 0
    }
}

TEST(RewardModel, PreferenceLossRanking)
{
    // Correct ranking (win > lose) must yield a lower Bradley-Terry loss.
    const float loss_correct = reward_preference_loss(2.0f, 1.0f);
    const float loss_wrong = reward_preference_loss(1.0f, 2.0f);
    const float loss_tie = reward_preference_loss(1.0f, 1.0f);

    EXPECT_GT(loss_correct, 0.0f);
    EXPECT_GT(loss_wrong, 0.0f);
    EXPECT_LT(loss_correct, loss_wrong);  // ranking matters
    EXPECT_NEAR(loss_tie, std::log(2.0f), 1e-5f);  // sigmoid(0) = 0.5
}

TEST(RewardModel, GenerateWithLogprobs)
{
    RewardModel rm(tiny_config());
    Tensor prompt({2});
    prompt[0] = 1.0f;
    prompt[1] = 2.0f;

    const auto result = rm.generate_with_logprobs(prompt, 4);
    EXPECT_EQ(result.log_probs.size(), 4u);
    EXPECT_EQ(result.tokens.shape()[0], prompt.shape()[0] + 4);
}

TEST(PPO, KlDivergence)
{
    const std::vector<float> log_p = {0.0f, -1.0f, -2.0f};
    const std::vector<float> log_q = {0.0f, -1.0f, -2.0f};
    EXPECT_FLOAT_EQ(kl_divergence(log_p, log_q), 0.0f);  // identical -> KL = 0

    const std::vector<float> log_p2 = {-0.5f, -0.5f};
    const std::vector<float> log_q2 = {-1.0f, -1.0f};
    const float kl = kl_divergence(log_p2, log_q2);
    EXPECT_GT(kl, 0.0f);  // different -> KL > 0
}

TEST(PPO, KlPenalty)
{
    const float penalized = kl_penalty(1.0f, 0.5f, 0.1f);
    EXPECT_FLOAT_EQ(penalized, 1.0f - 0.1f * 0.5f);
    // Positive KL reduces the reward
    EXPECT_LT(kl_penalty(1.0f, 1.0f, 0.1f), 1.0f);
}

TEST(PPO, MeanAndStdDev)
{
    const std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    EXPECT_FLOAT_EQ(mean(v), 3.0f);
    EXPECT_FLOAT_EQ(std_dev({5.0f, 5.0f, 5.0f}), 0.0f);
    EXPECT_GT(std_dev(v), 0.0f);
}

TEST(PPO, ComputeAdvantages)
{
    const std::vector<float> rewards = {1.0f, 2.0f, 3.0f};
    const std::vector<float> adv = compute_advantages(rewards);
    ASSERT_EQ(adv.size(), 3u);

    // Advantages are normalized: mean ~ 0
    EXPECT_NEAR(mean(adv), 0.0f, 1e-5f);
    // Higher reward -> higher advantage
    EXPECT_GT(adv[2], adv[0]);
}

TEST(PPO, ClippedObjective)
{
    const float adv = 1.0f;
    // Ratio within clip: objective = ratio * advantage
    EXPECT_FLOAT_EQ(ppo_clipped_objective(1.05f, adv, 0.2f), 1.05f);
    // Ratio above clip: objective = (1 + eps) * advantage
    EXPECT_FLOAT_EQ(ppo_clipped_objective(1.5f, adv, 0.2f), 1.2f);
    // Ratio below clip: min(ratio*A, clip*A) = ratio*A (the smaller one)
    EXPECT_FLOAT_EQ(ppo_clipped_objective(0.5f, adv, 0.2f), 0.5f);
}

TEST(PPO, PPOLoss)
{
    const std::vector<float> log_new = {0.0f, 0.0f};
    const std::vector<float> log_old = {0.0f, 0.0f};
    // ratio = 1 -> objective = advantage -> loss = -advantage
    EXPECT_FLOAT_EQ(ppo_loss(log_new, log_old, 1.0f, 0.2f), -1.0f);
}
