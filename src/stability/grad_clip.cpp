#include "grad_clip.h"

#include <cmath>
#include <limits>

namespace serius {
namespace stability {

float clip_grad_norm(std::vector<core::autograd::Node *> &params, float max_norm)
{
    // Step 1: compute global L2 norm of all gradients
    float total_norm_sq = 0.0f;
    for (auto *p : params) {
        if (p->grad.size() == 0) continue;
        const float *g = p->grad.data();
        size_t n = p->grad.size();
        for (size_t i = 0; i < n; i++) {
            total_norm_sq += g[i] * g[i];
        }
    }
    float total_norm = std::sqrt(total_norm_sq);

    // Step 2: clip if necessary
    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-6f);
        for (auto *p : params) {
            if (p->grad.size() == 0) continue;
            float *g = p->grad.data();
            size_t n = p->grad.size();
            for (size_t i = 0; i < n; i++) {
                g[i] *= scale;
            }
        }
    }

    return total_norm;
}

void clip_grad_value(std::vector<core::autograd::Node *> &params, float clip_value)
{
    for (auto *p : params) {
        if (p->grad.size() == 0) continue;
        float *g = p->grad.data();
        size_t n = p->grad.size();
        for (size_t i = 0; i < n; i++) {
            if (g[i] > clip_value) g[i] = clip_value;
            if (g[i] < -clip_value) g[i] = -clip_value;
        }
    }
}

bool has_nan_or_inf(const std::vector<core::autograd::Node *> &params)
{
    for (const auto *p : params) {
        if (p->grad.size() == 0) continue;
        const float *g = p->grad.data();
        size_t n = p->grad.size();
        for (size_t i = 0; i < n; i++) {
            if (std::isnan(g[i]) || std::isinf(g[i])) return true;
        }
    }
    return false;
}

}  // namespace stability
}  // namespace serius
