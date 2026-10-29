#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "../tensor/tensor.h"

namespace serius {
namespace liquid {

enum class ODESolverType
{
    EULER,
    RK4
};

using DerivativeFunc = std::function<core::Tensor(const core::Tensor &h, float t)>;

/// Neural Ordinary Differential Equation (Neural ODE) solver.
/// Evaluates continuous hidden state dynamics: dh/dt = f(h, t).
class NeuralODE
{
   public:
    /// Construct with a default parameterized linear + tanh neural derivative.
    explicit NeuralODE(size_t hidden_dim, ODESolverType solver = ODESolverType::RK4);

    /// Construct with a custom mathematical / neural derivative function.
    explicit NeuralODE(DerivativeFunc func, ODESolverType solver = ODESolverType::RK4);

    /// Integrate the hidden state from t0 to t1 using specified number of steps.
    core::Tensor forward(const core::Tensor &h0, float t0, float t1, int steps) const;

    /// Integrate and return the full trajectory of hidden states across all steps.
    std::vector<core::Tensor> forward_trajectory(const core::Tensor &h0, float t0, float t1, int steps) const;

    /// Single integration step using Euler method.
    core::Tensor step_euler(const core::Tensor &h, float t, float dt) const;

    /// Single integration step using 4th order Runge-Kutta (RK4).
    core::Tensor step_rk4(const core::Tensor &h, float t, float dt) const;

    /// Evaluate the derivative dh/dt at state h and time t.
    core::Tensor compute_derivative(const core::Tensor &h, float t) const;

    ODESolverType solver_type() const { return solver_; }
    void set_solver_type(ODESolverType s) { solver_ = s; }

    size_t hidden_dim() const { return hidden_dim_; }
    core::Tensor &weights() { return weights_; }
    const core::Tensor &weights() const { return weights_; }
    core::Tensor &bias() { return bias_; }
    const core::Tensor &bias() const { return bias_; }

   private:
    size_t hidden_dim_ = 0;
    ODESolverType solver_ = ODESolverType::RK4;
    DerivativeFunc custom_func_ = nullptr;
    core::Tensor weights_;  // [hidden_dim, hidden_dim]
    core::Tensor bias_;     // [hidden_dim]
};

}  // namespace liquid
}  // namespace serius
