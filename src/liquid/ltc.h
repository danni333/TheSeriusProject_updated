#pragma once

#include <cstddef>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace liquid {

/// Liquid Time-Constant (LTC) recurrent neural network layer.
/// Continuously adapts its time-constants tau(x) dynamically based on incoming inputs.
class LTCNetwork
{
   public:
    LTCNetwork(size_t in_features, size_t hidden_features,
               float tau_min = 0.1f, float tau_max = 5.0f);

    /// Compute state transition for a time interval dt.
    /// @param x   Input tensor of shape [in_features]
    /// @param h   Current hidden state tensor of shape [hidden_features]
    /// @param dt  Time increment (supports arbitrary irregular continuous time steps)
    /// @return Updated hidden state of shape [hidden_features]
    core::Tensor step(const core::Tensor &x, const core::Tensor &h, float dt) const;

    /// Compute dynamic time constants tau(x) for the given input.
    core::Tensor compute_tau(const core::Tensor &x) const;

    /// Compute target asymptotic state f(x, h) = tanh(W_h * h + W_x * x + b).
    core::Tensor compute_target(const core::Tensor &x, const core::Tensor &h) const;

    /// Process a sequence of inputs observed at continuous timestamps.
    /// @param inputs      Sequence of input tensors
    /// @param timestamps  Time markers corresponding to each input
    /// @param h           Initial state (modified in-place)
    /// @return Sequence of hidden states
    std::vector<core::Tensor> forward_sequence(const std::vector<core::Tensor> &inputs,
                                               const std::vector<float> &timestamps,
                                               core::Tensor &h) const;

    size_t in_features() const { return in_features_; }
    size_t hidden_features() const { return hidden_features_; }
    float tau_min() const { return tau_min_; }
    float tau_max() const { return tau_max_; }

    core::Tensor &W_x() { return W_x_; }
    const core::Tensor &W_x() const { return W_x_; }
    core::Tensor &W_h() { return W_h_; }
    const core::Tensor &W_h() const { return W_h_; }
    core::Tensor &b() { return b_; }
    const core::Tensor &b() const { return b_; }

    core::Tensor &W_tau() { return W_tau_; }
    const core::Tensor &W_tau() const { return W_tau_; }
    core::Tensor &b_tau() { return b_tau_; }
    const core::Tensor &b_tau() const { return b_tau_; }

   private:
    size_t in_features_;
    size_t hidden_features_;
    float tau_min_;
    float tau_max_;

    core::Tensor W_x_;    // [hidden, in]
    core::Tensor W_h_;    // [hidden, hidden]
    core::Tensor b_;      // [hidden]
    core::Tensor W_tau_;  // [hidden, in]
    core::Tensor b_tau_;  // [hidden]
};

}  // namespace liquid
}  // namespace serius
