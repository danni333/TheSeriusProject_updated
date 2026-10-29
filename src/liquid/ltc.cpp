#include "ltc.h"

#include <algorithm>
#include <cmath>

namespace serius {
namespace liquid {

namespace {

inline float sigmoid(float z)
{
    if (z >= 15.0f) return 1.0f;
    if (z <= -15.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-z));
}

}  // namespace

LTCNetwork::LTCNetwork(size_t in_features, size_t hidden_features, float tau_min, float tau_max)
    : in_features_(in_features),
      hidden_features_(hidden_features),
      tau_min_(tau_min),
      tau_max_(tau_max),
      W_x_({hidden_features, in_features}),
      W_h_({hidden_features, hidden_features}),
      b_({hidden_features}),
      W_tau_({hidden_features, in_features}),
      b_tau_({hidden_features})
{
    W_x_.zeros();
    W_h_.zeros();
    b_.zeros();
    W_tau_.zeros();
    b_tau_.zeros();

    // Initialize with stable baseline weights
    for (size_t i = 0; i < hidden_features_; ++i) {
        W_h_[i * hidden_features_ + i] = 0.5f;
        for (size_t j = 0; j < in_features_; ++j) {
            W_x_[i * in_features_ + j] = 0.2f;
            W_tau_[i * in_features_ + j] = 0.1f;
        }
    }
}

core::Tensor LTCNetwork::compute_tau(const core::Tensor &x) const
{
    core::Tensor tau({hidden_features_});
    float tau_range = tau_max_ - tau_min_;

    for (size_t i = 0; i < hidden_features_; ++i) {
        float sum = b_tau_[i];
        size_t in_limit = std::min(x.size(), in_features_);
        for (size_t j = 0; j < in_limit; ++j) {
            sum += W_tau_[i * in_features_ + j] * x[j];
        }
        tau[i] = tau_min_ + tau_range * sigmoid(sum);
    }

    return tau;
}

core::Tensor LTCNetwork::compute_target(const core::Tensor &x, const core::Tensor &h) const
{
    core::Tensor f({hidden_features_});

    for (size_t i = 0; i < hidden_features_; ++i) {
        float sum = b_[i];

        size_t h_limit = std::min(h.size(), hidden_features_);
        for (size_t j = 0; j < h_limit; ++j) {
            sum += W_h_[i * hidden_features_ + j] * h[j];
        }

        size_t x_limit = std::min(x.size(), in_features_);
        for (size_t j = 0; j < x_limit; ++j) {
            sum += W_x_[i * in_features_ + j] * x[j];
        }

        f[i] = std::tanh(sum);
    }

    return f;
}

core::Tensor LTCNetwork::step(const core::Tensor &x, const core::Tensor &h, float dt) const
{
    core::Tensor tau = compute_tau(x);
    core::Tensor target = compute_target(x, h);
    core::Tensor h_new({hidden_features_});

    float safe_dt = std::max(0.0f, dt);

    for (size_t i = 0; i < hidden_features_; ++i) {
        float t_val = std::max(1e-4f, tau[i]);
        // Exponential decay closed-form update: stable for any dt >= 0
        float decay = std::exp(-safe_dt / t_val);
        float h_prev = (i < h.size()) ? h[i] : 0.0f;
        h_new[i] = decay * h_prev + (1.0f - decay) * target[i];
    }

    return h_new;
}

std::vector<core::Tensor> LTCNetwork::forward_sequence(const std::vector<core::Tensor> &inputs,
                                                       const std::vector<float> &timestamps,
                                                       core::Tensor &h) const
{
    std::vector<core::Tensor> outputs;
    outputs.reserve(inputs.size());

    if (h.size() != hidden_features_) {
        h = core::Tensor({hidden_features_});
        h.zeros();
    }

    float prev_t = 0.0f;
    for (size_t k = 0; k < inputs.size(); ++k) {
        float dt = 1.0f;
        if (k < timestamps.size()) {
            float current_t = timestamps[k];
            dt = (k == 0) ? std::max(0.0f, current_t) : std::max(0.0f, current_t - prev_t);
            prev_t = current_t;
        }

        h = step(inputs[k], h, dt);
        outputs.push_back(h);
    }

    return outputs;
}

}  // namespace liquid
}  // namespace serius
