#include "circuit.h"

#include <cmath>

namespace serius {
namespace quantum {

namespace {
constexpr float kHalfPi = 1.5707963267948966f;  // pi / 2
}

QuantumCircuit::QuantumCircuit(size_t n_qubits, size_t n_layers)
    : n_qubits_(n_qubits), n_layers_(n_layers)
{
}

QuantumState QuantumCircuit::execute(const std::vector<float> &params) const
{
    QuantumState state(n_qubits_);
    size_t p = 0;

    for (size_t l = 0; l < n_layers_; ++l) {
        // Single-qubit parameterized rotations
        for (size_t q = 0; q < n_qubits_; ++q) {
            float theta = (p < params.size()) ? params[p++] : 0.0f;
            state.apply_gate(gates::RY(theta), q);
        }

        // Entangling ladder
        if (n_qubits_ > 1) {
            for (size_t q = 0; q + 1 < n_qubits_; ++q) {
                state.apply_cnot(q, q + 1);
            }
        }
    }

    return state;
}

float QuantumCircuit::evaluate(const std::vector<float> &params, const core::Tensor &cost_diag) const
{
    QuantumState state = execute(params);
    return state.expectation_diagonal(cost_diag);
}

std::vector<float> QuantumCircuit::parameter_shift_gradient(const std::vector<float> &params,
                                                            const core::Tensor &cost_diag) const
{
    size_t P = num_params();
    std::vector<float> gradients(P, 0.0f);
    std::vector<float> shifted = params;
    shifted.resize(P, 0.0f);

    for (size_t i = 0; i < P; ++i) {
        float original = shifted[i];

        // Shift forward by +pi/2
        shifted[i] = original + kHalfPi;
        float e_plus = evaluate(shifted, cost_diag);

        // Shift backward by -pi/2
        shifted[i] = original - kHalfPi;
        float e_minus = evaluate(shifted, cost_diag);

        // Parameter-shift exact derivative
        gradients[i] = 0.5f * (e_plus - e_minus);

        // Restore original
        shifted[i] = original;
    }

    return gradients;
}

}  // namespace quantum
}  // namespace serius
