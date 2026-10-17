#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../autograd/node.h"
#include "../autograd/ops.h"
#include "../common/serius_check.h"
#include "../tensor/tensor.h"

namespace serius {
namespace optim {

/**
 * @brief Automatic Mixed Precision (AMP) helper for training.
 *
 * This class handles dynamic loss scaling to prevent underflow in FP16/BF16 gradients.
 */
class AutomaticMixedPrecision
{
   public:
    static constexpr float DEFAULT_LOSS_SCALE = 65536.0F;
    static constexpr int DEFAULT_GROWTH_INTERVAL = 2000;
    static constexpr float DEFAULT_BACKOFF_FACTOR = 0.5F;
    static constexpr float DEFAULT_GROWTH_FACTOR = 2.0F;

    struct Config
    {
        float init_scale = 65536.0F;
        int growth_interval = 2000;
    };

    explicit AutomaticMixedPrecision(const Config &config = {})
        : current_scale_(config.init_scale), growth_interval_(config.growth_interval), steps_since_last_scale_(0)
    {
        if (!std::isfinite(current_scale_) || current_scale_ < 1.0F) {
            throw std::invalid_argument("AutomaticMixedPrecision: init_scale must be finite and at least 1.");
        }
        if (growth_interval_ <= 0) {
            throw std::invalid_argument("AutomaticMixedPrecision: growth_interval must be positive.");
        }
    }

    /**
     * @brief Scale the loss before the backward pass.
     * @param loss The loss node from the model forward pass.
     * @return A new scaled loss node.
     */
    core::autograd::Node *scale(core::autograd::Node *loss) const
    {
        return core::autograd::ops::mul_scalar(loss, current_scale_);
    }

    /**
     * @brief Unscale the gradients and check for NaNs/Infs.
     * @param parameters List of model parameters (nodes) to unscale gradients for.
     * @return True if gradients are finite (no overflow), False otherwise.
     */
    bool unscale_and_update(const std::vector<core::autograd::Node *> &parameters)
    {
        bool has_overflow = false;
        const float inv_scale = 1.0F / current_scale_;

        // Check for overflow (NaN or Inf) and unscale
        for (auto *parameter : parameters) {
            if (!parameter->requires_grad) {
                continue;
            }

            parameter->grad.syncToHost();
            for (size_t i = 0; i < parameter->grad.size(); ++i) {
                const float grad_val = parameter->grad.item(i);
                if (!std::isfinite(grad_val)) {
                    has_overflow = true;
                    break;
                }
            }
            if (has_overflow) {
                break;
            }
        }

        if (has_overflow) {
            // Back off: decrease scale and reset counter
            current_scale_ *= DEFAULT_BACKOFF_FACTOR;
            current_scale_ = std::max(current_scale_, 1.0F);
            steps_since_last_scale_ = 0;

            // Zero gradients to skip this step correctly
            for (auto *parameter : parameters) {
                if (parameter->requires_grad) {
                    parameter->grad.zeros();
                }
            }
            return false;
        }

        // No overflow: apply unscaling and increment growth counter
        for (auto *parameter : parameters) {
            if (!parameter->requires_grad) {
                continue;
            }

            // Apply unscaling on host (since we already synced for overflow check)
            for (size_t i = 0; i < parameter->grad.size(); ++i) {
                parameter->grad.set_item(i, parameter->grad.item(i) * inv_scale);
            }
            parameter->grad.mark_dirty(core::Tensor::DataLocation::CPU);
        }

        steps_since_last_scale_++;
        if (steps_since_last_scale_ >= growth_interval_) {
            current_scale_ = std::min(current_scale_ * DEFAULT_GROWTH_FACTOR,
                                      std::numeric_limits<float>::max());
            steps_since_last_scale_ = 0;
        }

        return true;
    }

    float current_scale() const { return current_scale_; }

   private:
    float current_scale_;
    int growth_interval_;
    int steps_since_last_scale_;
};

}  // namespace optim
}  // namespace serius
