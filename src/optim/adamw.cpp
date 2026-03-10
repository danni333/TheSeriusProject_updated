#include "adamw.h"

#include <cmath>

namespace serius {
namespace optim {

AdamW::AdamW(const std::vector<core::autograd::Node *> &params, const AdamWConfig &config)
    : params_(params), config_(config)
{
    for (auto *p : params_) {
        state_[p] = {core::Tensor::zeros(p->data.shape()), core::Tensor::zeros(p->data.shape()), 0};
    }
}

void AdamW::zero_grad()
{
    for (auto *p : params_) {
        if (p->requires_grad) {
            p->zero_grad();
        }
    }
}

void AdamW::step()
{
    for (auto *param_node : params_) {
        if (!param_node->requires_grad) {
            continue;
        }
        auto &adam_state = state_[param_node];
        adam_state.step++;

        const core::Tensor &grad_tensor = param_node->grad;
        core::Tensor &moment1 = adam_state.m;
        core::Tensor &moment2 = adam_state.v;
        core::Tensor &param_data = param_node->data;

        const float beta1 = config_.beta1;
        const float beta2 = config_.beta2;
        const float learning_rate = config_.lr;
        const float epsilon = config_.eps;
        const float weight_decay = config_.weight_decay;

        // Bias correction
        const float bias_correction1 = 1.0F - std::pow(beta1, static_cast<float>(adam_state.step));
        const float bias_correction2 = 1.0F - std::pow(beta2, static_cast<float>(adam_state.step));

// Use OpenMP for element-wise update if possible
#pragma omp parallel for
        for (long long elem_idx = 0; elem_idx < static_cast<long long>(param_data.size()); ++elem_idx) {
            const float grad_val = grad_tensor[elem_idx];

            // 1st moment update
            moment1[elem_idx] = (beta1 * moment1[elem_idx]) + ((1.0F - beta1) * grad_val);

            // 2nd moment update
            moment2[elem_idx] = (beta2 * moment2[elem_idx]) + ((1.0F - beta2) * (grad_val * grad_val));

            const float m_hat = moment1[elem_idx] / bias_correction1;
            const float v_hat = moment2[elem_idx] / bias_correction2;

            // Weight update with weight decay
            param_data[elem_idx] = param_data[elem_idx] - learning_rate * (m_hat / (std::sqrt(v_hat) + epsilon) +
                                                                           weight_decay * param_data[elem_idx]);
        }
    }
}

}  // namespace optim
}  // namespace serius
