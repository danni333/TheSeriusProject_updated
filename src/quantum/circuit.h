#pragma once

#include <cstddef>
#include <vector>

#include "../tensor/tensor.h"
#include "state.h"

namespace serius {
namespace quantum {

/// Parametric Quantum Circuit with hardware-efficient ansatz (RY rotations + entangling CNOT layers).
class QuantumCircuit
{
   public:
    QuantumCircuit(size_t n_qubits, size_t n_layers);

    /// Execute the parameterized circuit starting from ground state |0...0>.
    QuantumState execute(const std::vector<float> &params) const;

    /// Evaluate the expectation value of a diagonal Hamiltonian on the state produced by params.
    float evaluate(const std::vector<float> &params, const core::Tensor &cost_diag) const;

    /// Compute exact analytical gradients with respect to all circuit parameters
    /// using the Parameter-Shift Rule: d<H>/d_theta = (E(theta + pi/2) - E(theta - pi/2)) / 2.
    std::vector<float> parameter_shift_gradient(const std::vector<float> &params,
                                                const core::Tensor &cost_diag) const;

    size_t num_qubits() const { return n_qubits_; }
    size_t num_layers() const { return n_layers_; }
    size_t num_params() const { return n_qubits_ * n_layers_; }

   private:
    size_t n_qubits_;
    size_t n_layers_;
};

}  // namespace quantum
}  // namespace serius
