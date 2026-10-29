#pragma once

#include <cstddef>

#include "../tensor/tensor.h"

namespace serius {
namespace world {

/// Joint-Embedding Predictive Architecture (JEPA / V-JEPA).
/// Learns by predicting future representations in latent space rather than raw tokens/pixels.
class JEPA
{
   public:
    JEPA(size_t input_dim, size_t latent_dim, size_t action_dim = 0);

    /// Encode observation into latent representation using online encoder.
    core::Tensor encode(const core::Tensor &x) const;

    /// Encode observation into latent target using EMA target encoder.
    core::Tensor encode_target(const core::Tensor &x) const;

    /// Predict next latent representation given current latent state and action.
    core::Tensor predict_latent(const core::Tensor &z, const core::Tensor &action = core::Tensor()) const;

    /// Execute a training step on an observed transition (x_t, action, x_{t+1}).
    /// Returns the latent prediction MSE loss.
    float train_step(const core::Tensor &x_t, const core::Tensor &x_next,
                     const core::Tensor &action = core::Tensor(), float lr = 0.01f);

    /// Update target encoder weights using Exponential Moving Average (EMA).
    /// @param ema_decay Decay factor (e.g., 0.99f or 0.999f)
    void update_target_encoder(float ema_decay = 0.99f);

    size_t input_dim() const { return input_dim_; }
    size_t latent_dim() const { return latent_dim_; }
    size_t action_dim() const { return action_dim_; }

    core::Tensor &W_enc() { return W_enc_; }
    const core::Tensor &W_enc() const { return W_enc_; }
    core::Tensor &b_enc() { return b_enc_; }
    const core::Tensor &b_enc() const { return b_enc_; }

    core::Tensor &W_target() { return W_target_; }
    const core::Tensor &W_target() const { return W_target_; }
    core::Tensor &b_target() { return b_target_; }
    const core::Tensor &b_target() const { return b_target_; }

    core::Tensor &W_pred() { return W_pred_; }
    const core::Tensor &W_pred() const { return W_pred_; }

   private:
    size_t input_dim_;
    size_t latent_dim_;
    size_t action_dim_;

    // Online Encoder: maps input -> latent
    core::Tensor W_enc_;  // [latent_dim, input_dim]
    core::Tensor b_enc_;  // [latent_dim]

    // Target Encoder: EMA of online encoder
    core::Tensor W_target_;  // [latent_dim, input_dim]
    core::Tensor b_target_;  // [latent_dim]

    // Predictor: maps [latent, action] -> next latent
    core::Tensor W_pred_;  // [latent_dim, latent_dim + action_dim]
    core::Tensor b_pred_;  // [latent_dim]
};

}  // namespace world
}  // namespace serius
