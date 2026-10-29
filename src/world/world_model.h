#pragma once

#include <cstddef>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace world {

struct Prediction
{
    core::Tensor next_state;
    float predicted_reward = 0.0f;
    float uncertainty = 0.0f;
};

/// Predictive World Model in latent space.
/// Models the environment transition s_{t+1} = f(s_t, a_t) and predicts rewards and uncertainty.
class WorldModel
{
   public:
    WorldModel(size_t state_dim, size_t action_dim, size_t hidden_dim = 64);

    /// Predict next state, expected reward, and uncertainty given state and action.
    Prediction predict(const core::Tensor &state, const core::Tensor &action) const;

    /// Train transition and reward prediction on an observed transition (s, a) -> (s', r).
    float train_step(const core::Tensor &state, const core::Tensor &action,
                     const core::Tensor &target_next_state, float target_reward,
                     float lr = 0.01f);

    size_t state_dim() const { return state_dim_; }
    size_t action_dim() const { return action_dim_; }
    size_t hidden_dim() const { return hidden_dim_; }

    core::Tensor &W_trans() { return W_trans_; }
    const core::Tensor &W_trans() const { return W_trans_; }
    core::Tensor &b_trans() { return b_trans_; }
    const core::Tensor &b_trans() const { return b_trans_; }

    core::Tensor &W_state() { return W_state_; }
    const core::Tensor &W_state() const { return W_state_; }
    core::Tensor &b_state() { return b_state_; }
    const core::Tensor &b_state() const { return b_state_; }

    core::Tensor &W_reward() { return W_reward_; }
    const core::Tensor &W_reward() const { return W_reward_; }

   private:
    size_t state_dim_;
    size_t action_dim_;
    size_t hidden_dim_;

    // Layer 1: [state, action] -> hidden
    core::Tensor W_trans_;  // [hidden_dim, state_dim + action_dim]
    core::Tensor b_trans_;  // [hidden_dim]

    // Layer 2: hidden -> state_dim
    core::Tensor W_state_;  // [state_dim, hidden_dim]
    core::Tensor b_state_;  // [state_dim]

    // Reward head: state_dim -> scalar
    core::Tensor W_reward_;  // [state_dim]
    float b_reward_ = 0.0f;

    // Uncertainty head: hidden -> scalar
    core::Tensor W_unc_;  // [hidden_dim]
};

}  // namespace world
}  // namespace serius
