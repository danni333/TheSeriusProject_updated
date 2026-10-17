#pragma once

#include <vector>

#include "../autograd/node.h"

namespace serius {
namespace stability {

/// Dynamic Loss Scaler for mixed-precision training.
/// Scales the loss up before backward to prevent gradient underflow in FP16,
/// then scales gradients back down before the optimizer step.
class DynamicLossScaler
{
   public:
    struct Config
    {
        float initial_scale = 65536.0f;  // finite and at least 1
        float growth_factor = 2.0f;     // finite and greater than 1
        float backoff_factor = 0.5f;    // finite and in (0, 1)
        int growth_interval = 2000;     // positive steps without overflow before scaling up
    };

    DynamicLossScaler();
    explicit DynamicLossScaler(const Config &config);

    /// Get the current loss scale factor.
    float scale() const { return scale_; }

    /// Scale the loss value before backward pass.
    float scale_loss(float loss) const { return loss * scale_; }

    /// Unscale all gradients after backward pass (divide by scale).
    /// Returns false if NaN/Inf was found (step should be skipped).
    bool unscale_grads(std::vector<core::autograd::Node *> &params);

    /// Update the scale factor based on whether overflow occurred.
    /// Call this after unscale_grads.
    /// If overflow_detected is true, scale is reduced and step count resets.
    /// Otherwise, step count increments and scale may grow.
    void update(bool overflow_detected);

    // Accessors for monitoring
    int consecutive_good_steps() const { return good_steps_; }
    int overflow_count() const { return overflow_count_; }

   private:
    float scale_;
    float growth_factor_;
    float backoff_factor_;
    int growth_interval_;
    int good_steps_;
    int overflow_count_;
};

}  // namespace stability
}  // namespace serius
