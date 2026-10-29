#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../src/world/imagination.h"
#include "../src/world/jepa.h"
#include "../src/world/world_model.h"

using namespace serius::world;
using serius::core::Tensor;

// ---------------------------------------------------------------------------
// World Model Tests
// ---------------------------------------------------------------------------

TEST(WorldModelTest, PredictOutputDimensionsAndBounds)
{
    const size_t state_dim = 8;
    const size_t action_dim = 2;
    WorldModel wm(state_dim, action_dim, /*hidden_dim=*/16);

    Tensor s({state_dim});
    for (size_t i = 0; i < state_dim; ++i) s[i] = 0.5f;

    Tensor a({action_dim});
    a[0] = 1.0f; a[1] = -0.5f;

    Prediction pred = wm.predict(s, a);

    EXPECT_EQ(pred.next_state.size(), state_dim);
    EXPECT_TRUE(std::isfinite(pred.predicted_reward));
    EXPECT_GE(pred.uncertainty, 0.0f);
    EXPECT_LE(pred.uncertainty, 1.0f);

    for (size_t i = 0; i < state_dim; ++i) {
        EXPECT_TRUE(std::isfinite(pred.next_state[i]));
        EXPECT_GE(pred.next_state[i], -1.0f);
        EXPECT_LE(pred.next_state[i], 1.0f);
    }
}

TEST(WorldModelTest, TrainStepReducesPredictionLoss)
{
    const size_t state_dim = 4;
    const size_t action_dim = 2;
    WorldModel wm(state_dim, action_dim, /*hidden_dim=*/16);

    Tensor s({state_dim});
    for (size_t i = 0; i < state_dim; ++i) s[i] = 0.2f * static_cast<float>(i);

    Tensor a({action_dim});
    a[0] = 1.0f; a[1] = 0.0f;

    Tensor target_s({state_dim});
    for (size_t i = 0; i < state_dim; ++i) target_s[i] = 0.5f;
    float target_r = 1.0f;

    float initial_loss = wm.train_step(s, a, target_s, target_r, /*lr=*/0.05f);

    float final_loss = initial_loss;
    for (int step = 0; step < 25; ++step) {
        final_loss = wm.train_step(s, a, target_s, target_r, /*lr=*/0.05f);
    }

    EXPECT_LT(final_loss, initial_loss * 0.5f);
}

// ---------------------------------------------------------------------------
// Imagination Engine Tests
// ---------------------------------------------------------------------------

TEST(ImaginationTest, MultiStepRolloutStability)
{
    const size_t state_dim = 6;
    const size_t action_dim = 2;
    WorldModel wm(state_dim, action_dim, /*hidden_dim=*/16);
    ImaginationEngine engine;

    Tensor s0({state_dim});
    s0.zeros();

    const size_t horizon = 12;
    std::vector<Tensor> actions;
    for (size_t t = 0; t < horizon; ++t) {
        Tensor a({action_dim});
        a[0] = std::sin(static_cast<float>(t));
        a[1] = std::cos(static_cast<float>(t));
        actions.push_back(std::move(a));
    }

    TrajectoryRollout rollout = engine.imagine(wm, s0, actions, /*gamma=*/0.95f);

    ASSERT_EQ(rollout.steps.size(), horizon);
    EXPECT_TRUE(std::isfinite(rollout.cumulative_reward));
    EXPECT_GE(rollout.mean_uncertainty, 0.0f);
    EXPECT_LE(rollout.mean_uncertainty, 1.0f);

    for (const auto &step : rollout.steps) {
        for (size_t i = 0; i < state_dim; ++i) {
            EXPECT_TRUE(std::isfinite(step.next_state[i]));
            EXPECT_GE(step.next_state[i], -1.0f);
            EXPECT_LE(step.next_state[i], 1.0f);
        }
    }
}

TEST(ImaginationTest, PlanBestActionSequenceSelection)
{
    const size_t state_dim = 4;
    const size_t action_dim = 1;
    WorldModel wm(state_dim, action_dim, /*hidden_dim=*/8);

    // Bias reward head: positive states yield high reward
    for (size_t i = 0; i < state_dim; ++i) {
        wm.W_reward()[i] = 1.0f;
    }

    ImaginationEngine engine;
    Tensor s0({state_dim});
    s0.zeros();

    // Candidate 0: negative actions
    std::vector<Tensor> cand_neg;
    for (int t = 0; t < 5; ++t) {
        Tensor a({action_dim}); a[0] = -1.0f;
        cand_neg.push_back(std::move(a));
    }

    // Candidate 1: positive actions
    std::vector<Tensor> cand_pos;
    for (int t = 0; t < 5; ++t) {
        Tensor a({action_dim}); a[0] = 1.0f;
        cand_pos.push_back(std::move(a));
    }

    std::vector<std::vector<Tensor>> candidates = {cand_neg, cand_pos};
    int best = engine.plan_best_action_sequence(wm, s0, candidates);

    EXPECT_GE(best, 0);
    EXPECT_LE(best, 1);
}

// ---------------------------------------------------------------------------
// JEPA (Joint-Embedding Predictive Architecture) Tests
// ---------------------------------------------------------------------------

TEST(JEPATest, LatentLossDecreasesWithTraining)
{
    const size_t in_dim = 10;
    const size_t latent_dim = 4;
    JEPA jepa(in_dim, latent_dim);

    Tensor x_t({in_dim});
    Tensor x_next({in_dim});
    for (size_t i = 0; i < in_dim; ++i) {
        x_t[i] = 0.1f * static_cast<float>(i);
        x_next[i] = 0.15f * static_cast<float>(i + 1);
    }

    float initial_loss = jepa.train_step(x_t, x_next, /*action=*/Tensor(), /*lr=*/0.05f);

    float final_loss = initial_loss;
    for (int step = 0; step < 30; ++step) {
        final_loss = jepa.train_step(x_t, x_next, /*action=*/Tensor(), /*lr=*/0.05f);
    }

    EXPECT_LT(final_loss, initial_loss * 0.5f);
}

TEST(JEPATest, TargetEncoderEMAConvergence)
{
    const size_t in_dim = 4;
    const size_t latent_dim = 2;
    JEPA jepa(in_dim, latent_dim);

    // Modify online encoder weights
    for (size_t i = 0; i < jepa.W_enc().size(); ++i) {
        jepa.W_enc()[i] = 2.0f;
    }

    // Step EMA multiple times
    for (int step = 0; step < 50; ++step) {
        jepa.update_target_encoder(/*ema_decay=*/0.9f);
    }

    // Target encoder must smoothly approach online encoder
    for (size_t i = 0; i < jepa.W_target().size(); ++i) {
        EXPECT_NEAR(jepa.W_target()[i], 2.0f, 0.05f);
    }
}
