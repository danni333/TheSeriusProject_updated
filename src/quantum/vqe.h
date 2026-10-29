#pragma once

#include <cstddef>
#include <vector>

#include "../tensor/tensor.h"
#include "circuit.h"

namespace serius {
namespace quantum {

struct VQEResult
{
    std::vector<float> optimal_params;
    float minimum_energy = 0.0f;
    std::vector<float> energy_history;
    std::vector<float> final_probabilities;
    int iterations = 0;
};

/// Variational Quantum Eigensolver (VQE) and hybrid quantum-classical optimizer.
/// Solves combinatorial optimization and ground state problems using gradient descent
/// guided by the quantum parameter-shift rule.
class QuantumOptimizer
{
   public:
    QuantumOptimizer(size_t n_qubits, size_t n_layers);

    /// Run VQE optimization to find parameters that minimize the expectation of cost_diag.
    VQEResult optimize(const core::Tensor &cost_diag,
                       int max_iterations = 60,
                       float lr = 0.1f,
                       float tolerance = 1e-5f);

    float evaluate(const std::vector<float> &params, const core::Tensor &cost_diag) const;

    size_t num_qubits() const { return circuit_.num_qubits(); }
    size_t num_layers() const { return circuit_.num_layers(); }
    const QuantumCircuit &circuit() const { return circuit_; }

   private:
    QuantumCircuit circuit_;
};

}  // namespace quantum
}  // namespace serius
