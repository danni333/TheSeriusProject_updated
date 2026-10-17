#include "grad_clip.h"

#include <cmath>
#include <stdexcept>

namespace serius {
namespace stability {

float clip_grad_norm(const std::vector<core::autograd::Node *> &params, float max_norm)
{
    if (!std::isfinite(max_norm) || max_norm < 0.0F) {
        throw std::invalid_argument("clip_grad_norm: max_norm must be finite and non-negative");
    }

    // Step 1: compute global L2 norm of all gradients
    double total_norm_sq = 0.0;
    for (auto *p : params) {
        if (p->grad.size() == 0) continue;
        p->grad.syncToHost();
        const float *g = p->grad.data();
        size_t n = p->grad.size();
        for (size_t i = 0; i < n; i++) {
            if (!std::isfinite(g[i])) {
                throw std::runtime_error("clip_grad_norm: gradient contains NaN or Inf");
            }
            const double value = static_cast<double>(g[i]);
            total_norm_sq += value * value;
        }
    }
    const double total_norm = std::sqrt(total_norm_sq);

    // Step 2: clip if necessary
    if (total_norm > max_norm) {
        const float scale = static_cast<float>(static_cast<double>(max_norm) / total_norm);
        for (auto *p : params) {
            if (p->grad.size() == 0) continue;
            float *g = p->grad.data();
            size_t n = p->grad.size();
            for (size_t i = 0; i < n; i++) {
                g[i] *= scale;
            }
            p->grad.mark_dirty(core::Tensor::DataLocation::CPU);
        }
    }

    return static_cast<float>(total_norm);
}

void clip_grad_value(std::vector<core::autograd::Node *> &params, float clip_value)
{
    if (!std::isfinite(clip_value) || clip_value < 0.0F) {
        throw std::invalid_argument("clip_grad_value: clip_value must be finite and non-negative");
    }

    for (auto *p : params) {
        if (p->grad.size() == 0) continue;
        p->grad.syncToHost();
        float *g = p->grad.data();
        size_t n = p->grad.size();
        for (size_t i = 0; i < n; i++) {
            if (g[i] > clip_value) g[i] = clip_value;
            if (g[i] < -clip_value) g[i] = -clip_value;
        }
        p->grad.mark_dirty(core::Tensor::DataLocation::CPU);
    }
}

bool has_nan_or_inf(const std::vector<core::autograd::Node *> &params)
{
    for (auto *p : params) {
        if (p->grad.size() == 0) continue;
        p->grad.syncToHost();
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
