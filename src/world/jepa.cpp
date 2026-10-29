#include "jepa.h"

#include <algorithm>
#include <cmath>

namespace serius {
namespace world {

JEPA::JEPA(size_t input_dim, size_t latent_dim, size_t action_dim)
    : input_dim_(input_dim),
      latent_dim_(latent_dim),
      action_dim_(action_dim),
      W_enc_({latent_dim, input_dim}),
      b_enc_({latent_dim}),
      W_target_({latent_dim, input_dim}),
      b_target_({latent_dim}),
      W_pred_({latent_dim, latent_dim + action_dim}),
      b_pred_({latent_dim})
{
    W_enc_.zeros();
    b_enc_.zeros();
    W_target_.zeros();
    b_target_.zeros();
    W_pred_.zeros();
    b_pred_.zeros();

    float scale_enc = 1.0f / std::sqrt(static_cast<float>(input_dim_));
    for (size_t i = 0; i < latent_dim_; ++i) {
        for (size_t j = 0; j < input_dim_; ++j) {
            float val = ((i + j) % 2 == 0 ? 1.0f : -1.0f) * scale_enc * 0.5f;
            W_enc_[i * input_dim_ + j] = val;
            W_target_[i * input_dim_ + j] = val;
        }
    }

    size_t pred_in_dim = latent_dim_ + action_dim_;
    for (size_t i = 0; i < latent_dim_; ++i) {
        W_pred_[i * pred_in_dim + i] = 0.8f;  // Near identity propagation in latent space
    }
}

core::Tensor JEPA::encode(const core::Tensor &x) const
{
    core::Tensor z({latent_dim_});
    for (size_t i = 0; i < latent_dim_; ++i) {
        float sum = b_enc_[i];
        size_t limit = std::min(x.size(), input_dim_);
        for (size_t j = 0; j < limit; ++j) {
            sum += W_enc_[i * input_dim_ + j] * x[j];
        }
        z[i] = std::tanh(sum);
    }
    return z;
}

core::Tensor JEPA::encode_target(const core::Tensor &x) const
{
    core::Tensor z({latent_dim_});
    for (size_t i = 0; i < latent_dim_; ++i) {
        float sum = b_target_[i];
        size_t limit = std::min(x.size(), input_dim_);
        for (size_t j = 0; j < limit; ++j) {
            sum += W_target_[i * input_dim_ + j] * x[j];
        }
        z[i] = std::tanh(sum);
    }
    return z;
}

core::Tensor JEPA::predict_latent(const core::Tensor &z, const core::Tensor &action) const
{
    size_t pred_in_dim = latent_dim_ + action_dim_;
    std::vector<float> input(pred_in_dim, 0.0f);

    size_t z_limit = std::min(z.size(), latent_dim_);
    for (size_t i = 0; i < z_limit; ++i) {
        input[i] = z[i];
    }

    size_t a_limit = std::min(action.size(), action_dim_);
    for (size_t i = 0; i < a_limit; ++i) {
        input[latent_dim_ + i] = action[i];
    }

    core::Tensor z_pred({latent_dim_});
    for (size_t i = 0; i < latent_dim_; ++i) {
        float sum = b_pred_[i];
        for (size_t j = 0; j < pred_in_dim; ++j) {
            sum += W_pred_[i * pred_in_dim + j] * input[j];
        }
        z_pred[i] = std::tanh(sum);
    }

    return z_pred;
}

float JEPA::train_step(const core::Tensor &x_t, const core::Tensor &x_next,
                       const core::Tensor &action, float lr)
{
    core::Tensor z_t = encode(x_t);
    core::Tensor z_pred = predict_latent(z_t, action);
    core::Tensor z_target = encode_target(x_next);

    float loss = 0.0f;
    size_t pred_in_dim = latent_dim_ + action_dim_;
    std::vector<float> d_pred(latent_dim_, 0.0f);

    for (size_t i = 0; i < latent_dim_; ++i) {
        float diff = z_pred[i] - z_target[i];
        loss += diff * diff;
        d_pred[i] = 2.0f * diff / static_cast<float>(latent_dim_);
    }
    loss /= static_cast<float>(latent_dim_);

    // Gradient descent on predictor
    for (size_t i = 0; i < latent_dim_; ++i) {
        float dtanh = 1.0f - z_pred[i] * z_pred[i];
        float grad_out = d_pred[i] * dtanh;

        for (size_t j = 0; j < latent_dim_; ++j) {
            float in_val = (j < z_t.size()) ? z_t[j] : 0.0f;
            W_pred_[i * pred_in_dim + j] -= lr * grad_out * in_val;
        }

        for (size_t j = 0; j < action_dim_; ++j) {
            float in_val = (j < action.size()) ? action[j] : 0.0f;
            W_pred_[i * pred_in_dim + latent_dim_ + j] -= lr * grad_out * in_val;
        }

        b_pred_[i] -= lr * grad_out;
    }

    // Gradient descent on online encoder
    for (size_t i = 0; i < latent_dim_; ++i) {
        float dtanh = 1.0f - z_t[i] * z_t[i];
        float grad = d_pred[i] * dtanh;
        size_t limit = std::min(x_t.size(), input_dim_);
        for (size_t j = 0; j < limit; ++j) {
            W_enc_[i * input_dim_ + j] -= lr * grad * x_t[j];
        }
        b_enc_[i] -= lr * grad;
    }

    return loss;
}

void JEPA::update_target_encoder(float ema_decay)
{
    float decay = std::max(0.0f, std::min(1.0f, ema_decay));
    float one_minus_decay = 1.0f - decay;

    for (size_t i = 0; i < W_target_.size(); ++i) {
        W_target_[i] = decay * W_target_[i] + one_minus_decay * W_enc_[i];
    }

    for (size_t i = 0; i < b_target_.size(); ++i) {
        b_target_[i] = decay * b_target_[i] + one_minus_decay * b_enc_[i];
    }
}

}  // namespace world
}  // namespace serius
