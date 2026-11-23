#include "vqe.h"

#include <cmath>

namespace serius {
namespace quantum {

QuantumOptimizer::QuantumOptimizer(size_t n_qubits, size_t n_layers)
    : circuit_(n_qubits, n_layers)
{
}

float QuantumOptimizer::evaluate(const std::vector<float> &params, const core::Tensor &cost_diag) const
{
    return circuit_.evaluate(params, cost_diag);
}

VQEResult QuantumOptimizer::optimize(const core::Tensor &cost_diag,
                                     int max_iterations,
                                     float lr,
                                     float tolerance)
{
    VQEResult result;
    size_t P = circuit_.num_params();
    std::vector<float> params(P, 0.0f);

    // Small initial perturbation
    for (size_t i = 0; i < P; ++i) {
        params[i] = 0.1f * ((i % 2 == 0) ? 1.0f : -1.0f);
    }

    float prev_energy = evaluate(params, cost_diag);
    result.energy_history.push_back(prev_energy);

    int iter = 0;
    for (; iter < max_iterations; ++iter) {
        std::vector<float> grads = circuit_.parameter_shift_gradient(params, cost_diag);

        for (size_t i = 0; i < P; ++i) {
            params[i] -= lr * grads[i];
        }

        float current_energy = evaluate(params, cost_diag);
        result.energy_history.push_back(current_energy);

        if (std::abs(prev_energy - current_energy) < tolerance) {
            prev_energy = current_energy;
            iter++;
            break;
        }

        prev_energy = current_energy;
    }

    result.optimal_params = params;
    result.minimum_energy = prev_energy;
    result.iterations = iter;

    QuantumState final_state = circuit_.execute(params);
    result.final_probabilities = final_state.probabilities();

    return result;
}

}  // namespace quantum
}  // namespace serius
