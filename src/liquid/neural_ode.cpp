#include "neural_ode.h"

#include <cmath>
#include <stdexcept>

namespace serius {
namespace liquid {

namespace {

core::Tensor add_tensors(const core::Tensor &a, const core::Tensor &b)
{
    core::Tensor out(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

core::Tensor scale_add(const core::Tensor &a, float scale, const core::Tensor &b)
{
    core::Tensor out(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + scale * b[i];
    }
    return out;
}

}  // namespace

NeuralODE::NeuralODE(size_t hidden_dim, ODESolverType solver)
    : hidden_dim_(hidden_dim),
      solver_(solver),
      weights_({hidden_dim, hidden_dim}),
      bias_({hidden_dim})
{
    weights_.zeros();
    bias_.zeros();
    // Default to stable contractive dynamics
    for (size_t i = 0; i < hidden_dim_; ++i) {
        weights_[i * hidden_dim_ + i] = -0.5f;
    }
}

NeuralODE::NeuralODE(DerivativeFunc func, ODESolverType solver)
    : hidden_dim_(0), solver_(solver), custom_func_(std::move(func))
{
}

core::Tensor NeuralODE::compute_derivative(const core::Tensor &h, float t) const
{
    if (custom_func_) {
        return custom_func_(h, t);
    }

    // Default parameterized derivative: f(h) = tanh(W * h + b)
    core::Tensor dh(h.shape());
    dh.zeros();
    for (size_t i = 0; i < hidden_dim_; ++i) {
        float sum = bias_[i];
        for (size_t j = 0; j < hidden_dim_; ++j) {
            sum += weights_[i * hidden_dim_ + j] * h[j];
        }
        dh[i] = std::tanh(sum);
    }
    return dh;
}

core::Tensor NeuralODE::step_euler(const core::Tensor &h, float t, float dt) const
{
    core::Tensor k1 = compute_derivative(h, t);
    return scale_add(h, dt, k1);
}

core::Tensor NeuralODE::step_rk4(const core::Tensor &h, float t, float dt) const
{
    // k1 = f(h, t)
    core::Tensor k1 = compute_derivative(h, t);

    // k2 = f(h + 0.5 * dt * k1, t + 0.5 * dt)
    core::Tensor h_k2 = scale_add(h, 0.5f * dt, k1);
    core::Tensor k2 = compute_derivative(h_k2, t + 0.5f * dt);

    // k3 = f(h + 0.5 * dt * k2, t + 0.5 * dt)
    core::Tensor h_k3 = scale_add(h, 0.5f * dt, k2);
    core::Tensor k3 = compute_derivative(h_k3, t + 0.5f * dt);

    // k4 = f(h + dt * k3, t + dt)
    core::Tensor h_k4 = scale_add(h, dt, k3);
    core::Tensor k4 = compute_derivative(h_k4, t + dt);

    // h_next = h + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    core::Tensor h_next(h.shape());
    float factor = dt / 6.0f;
    for (size_t i = 0; i < h.size(); ++i) {
        h_next[i] = h[i] + factor * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    }

    return h_next;
}

core::Tensor NeuralODE::forward(const core::Tensor &h0, float t0, float t1, int steps) const
{
    if (steps <= 0) {
        return h0;
    }

    core::Tensor h = h0;
    float dt = (t1 - t0) / static_cast<float>(steps);

    for (int i = 0; i < steps; ++i) {
        float t = t0 + static_cast<float>(i) * dt;
        if (solver_ == ODESolverType::RK4) {
            h = step_rk4(h, t, dt);
        } else {
            h = step_euler(h, t, dt);
        }
    }

    return h;
}

std::vector<core::Tensor> NeuralODE::forward_trajectory(const core::Tensor &h0, float t0, float t1, int steps) const
{
    std::vector<core::Tensor> trajectory;
    trajectory.reserve(steps > 0 ? steps + 1 : 1);
    trajectory.push_back(h0);

    if (steps <= 0) {
        return trajectory;
    }

    core::Tensor h = h0;
    float dt = (t1 - t0) / static_cast<float>(steps);

    for (int i = 0; i < steps; ++i) {
        float t = t0 + static_cast<float>(i) * dt;
        if (solver_ == ODESolverType::RK4) {
            h = step_rk4(h, t, dt);
        } else {
            h = step_euler(h, t, dt);
        }
        trajectory.push_back(h);
    }

    return trajectory;
}

}  // namespace liquid
}  // namespace serius
