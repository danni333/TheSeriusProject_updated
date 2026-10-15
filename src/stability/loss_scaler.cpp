#include "loss_scaler.h"

#include <cmath>

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
}

bool DynamicLossScaler::unscale_grads(std::vector<core::autograd::Node *> &params)
{
    float inv_scale = 1.0f / scale_;
    bool found_inf = false;

    for (auto *p : params) {
        if (p->grad.size() == 0) continue;
        float *g = p->grad.data();
        size_t n = p->grad.size();
        for (size_t i = 0; i < n; i++) {
            g[i] *= inv_scale;
            if (std::isnan(g[i]) || std::isinf(g[i])) {
                found_inf = true;
            }
        }
    }

    return !found_inf;  // returns true if grads are OK (no overflow)
}

void DynamicLossScaler::update(bool overflow_detected)
{
    if (overflow_detected) {
        // Reduce scale
        scale_ *= backoff_factor_;
        good_steps_ = 0;
        overflow_count_++;
    } else {
        good_steps_++;
        if (good_steps_ >= growth_interval_) {
            scale_ *= growth_factor_;
            good_steps_ = 0;
        }
    }
}

}  // namespace stability
}  // namespace serius
