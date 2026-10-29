#include "adaptive.h"

#include <algorithm>
#include <cmath>

namespace serius {
namespace liquid {

AdaptiveStepController::AdaptiveStepController(float tolerance, int min_steps, int max_steps)
    : tolerance_(tolerance), min_steps_(min_steps), max_steps_(max_steps)
{
}

float AdaptiveStepController::estimate_complexity(const core::Tensor &x)
{
    if (x.size() <= 1) {
        return 0.1f;
    }

    // Mean
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i];
    }
    float mean = sum / static_cast<float>(x.size());

    // Variance
    float var = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= static_cast<float>(x.size());

    // High-frequency gradient/fluctuation
    float tv = 0.0f;
    for (size_t i = 1; i < x.size(); ++i) {
        tv += std::abs(x[i] - x[i - 1]);
    }
    tv /= static_cast<float>(x.size() - 1);

    // Normalized complexity in [0, 1]
    float raw = 0.5f * std::sqrt(var) + 0.5f * tv;
    return std::max(0.0f, std::min(1.0f, raw / (1.0f + raw)));
}

int AdaptiveStepController::compute_steps(const core::Tensor &x) const
{
    float c = estimate_complexity(x);
    int step_range = max_steps_ - min_steps_;
    int steps = min_steps_ + static_cast<int>(std::round(c * static_cast<float>(step_range)));
    return std::max(min_steps_, std::min(max_steps_, steps));
}

int AdaptiveStepController::solve_until_convergence(const LTCNetwork &ltc, const core::Tensor &x,
                                                    core::Tensor &h, float tolerance) const
{
    float tol = (tolerance > 0.0f) ? tolerance : tolerance_;

    if (h.size() != ltc.hidden_features()) {
        h = core::Tensor({ltc.hidden_features()});
        h.zeros();
    }

    for (int steps = 1; steps <= max_steps_; ++steps) {
        float dt = 1.0f / static_cast<float>(steps);
        core::Tensor h_new = ltc.step(x, h, dt);

        // L2 distance between h_new and h
        float dist_sq = 0.0f;
        for (size_t i = 0; i < h.size(); ++i) {
            float d = h_new[i] - h[i];
            dist_sq += d * d;
        }

        h = h_new;

        if (std::sqrt(dist_sq) < tol && steps >= min_steps_) {
            return steps;
        }
    }

    return max_steps_;
}

}  // namespace liquid
}  // namespace serius
