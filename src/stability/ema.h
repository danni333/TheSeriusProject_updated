#pragma once

#include <vector>

#include "../autograd/node.h"

namespace serius {
namespace stability {

/// Exponential Moving Average (EMA) of model weights.
/// Maintains shadow copies of parameters that track a smoothed version.
/// The EMA model typically generalizes better on validation/inference.
class EMAWeights
{
   public:
    /// Create EMA tracker with given decay rate (typically 0.999 or 0.9999).
    explicit EMAWeights(float decay = 0.999f);

    /// Initialize shadow parameters from current model parameters.
    /// Must be called once before update().
    void init(const std::vector<core::autograd::Node *> &model_params);

    /// Update shadow parameters: shadow = decay * shadow + (1 - decay) * current
    void update(const std::vector<core::autograd::Node *> &model_params);

    /// Copy shadow parameters into the model (for evaluation/inference).
    /// Saves the original model weights internally so they can be restored.
    void apply(std::vector<core::autograd::Node *> &model_params);

    /// Restore original model weights (undo apply).
    void restore(std::vector<core::autograd::Node *> &model_params);

    /// Get the number of shadow parameters.
    size_t num_params() const { return shadow_params_.size(); }

    /// Access shadow parameter by index (read-only).
    const core::Tensor &shadow(size_t idx) const { return shadow_params_[idx]; }

    float decay() const { return decay_; }
    void set_decay(float d) { decay_ = d; }

   private:
    float decay_;
    std::vector<core::Tensor> shadow_params_;
    std::vector<core::Tensor> backup_params_;  // saved originals during apply()
    bool applied_ = false;
};

}  // namespace stability
}  // namespace serius
