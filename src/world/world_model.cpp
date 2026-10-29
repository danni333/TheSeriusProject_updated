#include "world_model.h"

#include <algorithm>
#include <cmath>

namespace serius {
namespace world {

WorldModel::WorldModel(size_t state_dim, size_t action_dim, size_t hidden_dim)
    : state_dim_(state_dim),
      action_dim_(action_dim),
      hidden_dim_(hidden_dim),
      W_trans_({hidden_dim, state_dim + action_dim}),
      b_trans_({hidden_dim}),
      W_state_({state_dim, hidden_dim}),
      b_state_({state_dim}),
      W_reward_({state_dim}),
      b_reward_(0.0f),
      W_unc_({hidden_dim})
{
    W_trans_.zeros();
    b_trans_.zeros();
    W_state_.zeros();
    b_state_.zeros();
    W_reward_.zeros();
    W_unc_.zeros();

    size_t in_dim = state_dim_ + action_dim_;

    // Initialize with stable identity + residual behavior
    float scale1 = 1.0f / std::sqrt(static_cast<float>(in_dim));
    for (size_t i = 0; i < hidden_dim_; ++i) {
        for (size_t j = 0; j < in_dim; ++j) {
            W_trans_[i * in_dim + j] = ((i + j) % 2 == 0 ? 1.0f : -1.0f) * scale1 * 0.5f;
        }
        W_unc_[i] = 0.05f;
    }

    float scale2 = 1.0f / std::sqrt(static_cast<float>(hidden_dim_));
    for (size_t i = 0; i < state_dim_; ++i) {
        for (size_t j = 0; j < hidden_dim_; ++j) {
            W_state_[i * hidden_dim_ + j] = ((i + j) % 2 == 0 ? 1.0f : -1.0f) * scale2 * 0.5f;
        }
        W_reward_[i] = 0.1f;
    }
}

Prediction WorldModel::predict(const core::Tensor &state, const core::Tensor &action) const
{
    size_t in_dim = state_dim_ + action_dim_;
    std::vector<float> input(in_dim, 0.0f);

    size_t s_limit = std::min(state.size(), state_dim_);
    for (size_t i = 0; i < s_limit; ++i) {
        input[i] = state[i];
    }

    size_t a_limit = std::min(action.size(), action_dim_);
    for (size_t i = 0; i < a_limit; ++i) {
        input[state_dim_ + i] = action[i];
    }

    // Layer 1: hidden = tanh(W_trans * [state, action] + b_trans)
    std::vector<float> hidden(hidden_dim_, 0.0f);
    for (size_t i = 0; i < hidden_dim_; ++i) {
        float sum = b_trans_[i];
        for (size_t j = 0; j < in_dim; ++j) {
            sum += W_trans_[i * in_dim + j] * input[j];
        }
        hidden[i] = std::tanh(sum);
    }

    // Layer 2: next_state = residual state + W_state * hidden + b_state
    core::Tensor next_state({state_dim_});
    for (size_t i = 0; i < state_dim_; ++i) {
        float sum = b_state_[i];
        for (size_t j = 0; j < hidden_dim_; ++j) {
            sum += W_state_[i * hidden_dim_ + j] * hidden[j];
        }
        float base = (i < state.size()) ? state[i] : 0.0f;
        next_state[i] = std::tanh(base + sum);
    }

    // Reward head: reward = sum(W_reward * next_state) + b_reward
    float reward = b_reward_;
    for (size_t i = 0; i < state_dim_; ++i) {
        reward += W_reward_[i] * next_state[i];
    }

    // Uncertainty head: variance proxy = sigmoid(W_unc * hidden)
    float unc_sum = 0.0f;
    for (size_t i = 0; i < hidden_dim_; ++i) {
        unc_sum += W_unc_[i] * hidden[i];
    }
    float unc = 1.0f / (1.0f + std::exp(-unc_sum));

    return Prediction{next_state, reward, unc};
}

float WorldModel::train_step(const core::Tensor &state, const core::Tensor &action,
                             const core::Tensor &target_next_state, float target_reward,
                             float lr)
{
    Prediction pred = predict(state, action);

    // Compute MSE loss for next state
    float state_loss = 0.0f;
    std::vector<float> d_state(state_dim_, 0.0f);
    for (size_t i = 0; i < state_dim_; ++i) {
        float target_val = (i < target_next_state.size()) ? target_next_state[i] : 0.0f;
        float err = pred.next_state[i] - target_val;
        state_loss += err * err;
        d_state[i] = 2.0f * err / static_cast<float>(state_dim_);
    }
    state_loss /= static_cast<float>(state_dim_);

    // Compute MSE loss for reward
    float r_err = pred.predicted_reward - target_reward;
    float reward_loss = r_err * r_err;
    float d_reward = 2.0f * r_err;

    // Gradient update on W_reward and b_reward
    for (size_t i = 0; i < state_dim_; ++i) {
        W_reward_[i] -= lr * d_reward * pred.next_state[i];
    }
    b_reward_ -= lr * d_reward;

    // Gradient update on W_state
    for (size_t i = 0; i < state_dim_; ++i) {
        for (size_t j = 0; j < hidden_dim_; ++j) {
            W_state_[i * hidden_dim_ + j] -= lr * d_state[i];
        }
        b_state_[i] -= lr * d_state[i];
    }

    return state_loss + reward_loss;
}

}  // namespace world
}  // namespace serius
