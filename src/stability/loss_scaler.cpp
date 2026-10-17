#include "loss_scaler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace serius {
namespace stability {

DynamicLossScaler::DynamicLossScaler() : DynamicLossScaler(Config()) {}

DynamicLossScaler::DynamicLossScaler(const Config &config)
    : scale_(config.initial_scale),
      growth_factor_(config.growth_factor),
      backoff_factor_(config.backoff_factor),
      growth_interval_(config.growth_interval),
      good_steps_(0),
      overflow_count_(0)
{
    if (!std::isfinite(scale_) || scale_ < 1.0f) {
        throw std::invalid_argument("DynamicLossScaler: initial_scale must be finite and at least 1.");
    }
    if (!std::isfinite(growth_factor_) || growth_factor_ <= 1.0f) {
        throw std::invalid_argument("DynamicLossScaler: growth_factor must be finite and greater than 1.");
    }
    if (!std::isfinite(backoff_factor_) || backoff_factor_ <= 0.0f || backoff_factor_ >= 1.0f) {
        throw std::invalid_argument("DynamicLossScaler: backoff_factor must be finite and between 0 and 1.");
    }
    if (growth_interval_ <= 0) {
        throw std::invalid_argument("DynamicLossScaler: growth_interval must be greater than 0.");
    }
}

bool DynamicLossScaler::unscale_grads(std::vector<core::autograd::Node *> &params)
{
    float inv_scale = 1.0f / scale_;
    bool found_inf = false;

    for (auto *p : params) {
        if (p->grad.size() == 0) continue;
        p->grad.syncToHost();
        float *g = p->grad.data();
        size_t n = p->grad.size();
        for (size_t i = 0; i < n; i++) {
            g[i] *= inv_scale;
            if (std::isnan(g[i]) || std::isinf(g[i])) {
                found_inf = true;
            }
        }
        p->grad.mark_dirty(core::Tensor::DataLocation::CPU);
    }

    return !found_inf;  // returns true if grads are OK (no overflow)
}

void DynamicLossScaler::update(bool overflow_detected)
{
    if (overflow_detected) {
        // Reduce scale
        scale_ = std::max(1.0f, scale_ * backoff_factor_);
        good_steps_ = 0;
        overflow_count_++;
    } else {
        good_steps_++;
        if (good_steps_ >= growth_interval_) {
            scale_ = std::min(scale_ * growth_factor_,
                              std::numeric_limits<float>::max());
            good_steps_ = 0;
        }
    }
}

}  // namespace stability
}  // namespace serius
